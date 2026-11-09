#if canImport(Network)
  import Foundation
  import Network

  /// A local HTTP server that answers OpenAI embeddings requests with fixed vectors.
  ///
  /// The server listens on a free port of the loopback interface,
  /// reads one request on each connection, answers it, and closes the connection.
  final class MockEmbeddingServer: @unchecked Sendable {
    /// What the server sends back.
    enum Behavior: Sendable {
      /// Vectors of `dimension` elements, in reverse order of the inputs.
      case vectors(dimension: Int)

      /// An HTTP status with an OpenAI error body.
      case error(status: Int, message: String)

      /// One vector fewer than the inputs.
      case missingVector(dimension: Int)

      /// A JSON body with status 200, for any request.
      case json(String)

      /// A plain-text body with the status.
      case text(status: Int, body: String)
    }

    /// A request that the server received.
    struct Received: Sendable {
      var path: String
      var headers: [String: String]
      var body: [String: any Sendable]
    }

    private let listener: NWListener
    private let queue = DispatchQueue(label: "MockEmbeddingServer")
    private let lock = NSLock()
    private var behavior: Behavior
    private var received: [Received] = []

    init(behavior: Behavior) throws {
      self.behavior = behavior
      let parameters = NWParameters.tcp
      parameters.requiredLocalEndpoint = NWEndpoint.hostPort(host: .ipv4(.loopback), port: .any)
      listener = try NWListener(using: parameters)
    }

    /// Starts the server and returns its base URL, such as `http://127.0.0.1:52011/v1`.
    func start() async throws -> URL {
      let port: UInt16 = try await withCheckedThrowingContinuation { continuation in
        let resumed = TestLock(false)
        // Returns `true` the first time only, so that the continuation resumes once.
        let claim: @Sendable () -> Bool = {
          resumed.withLock { done in
            if done { return false }
            done = true
            return true
          }
        }
        listener.stateUpdateHandler = { [listener] state in
          switch state {
          case .ready:
            if claim() { continuation.resume(returning: listener.port?.rawValue ?? 0) }
          case .failed(let error), .waiting(let error):
            if claim() { continuation.resume(throwing: error) }
          default:
            break
          }
        }
        listener.newConnectionHandler = { [weak self] connection in
          self?.handle(connection)
        }
        listener.start(queue: queue)
      }
      return URL(string: "http://127.0.0.1:\(port)/v1")!
    }

    func stop() {
      listener.cancel()
    }

    /// Changes what the server sends to later requests.
    func setBehavior(_ behavior: Behavior) {
      lock.withLock { self.behavior = behavior }
    }

    var requests: [Received] {
      lock.withLock { received }
    }

    private func handle(_ connection: NWConnection) {
      connection.start(queue: queue)
      read(connection, buffer: Data())
    }

    /// Reads until the buffer has the headers and the whole body, then answers.
    private func read(_ connection: NWConnection, buffer: Data) {
      connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) {
        [weak self] data, _, isComplete, error in
        guard let self else { return }
        var buffer = buffer
        if let data { buffer.append(data) }
        if let request = Self.parse(buffer) {
          self.respond(to: request, on: connection)
        } else if isComplete || error != nil {
          connection.cancel()
        } else {
          self.read(connection, buffer: buffer)
        }
      }
    }

    /// Returns the request in `buffer`, or `nil` if the buffer doesn't have all of it yet.
    private static func parse(_ buffer: Data) -> Received? {
      guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else { return nil }
      let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self)
      var lines = head.components(separatedBy: "\r\n")
      let requestLine = lines.removeFirst().split(separator: " ")
      var headers: [String: String] = [:]
      for line in lines {
        guard let colon = line.firstIndex(of: ":") else { continue }
        let name = line[..<colon].lowercased()
        headers[name] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
      }
      let length = Int(headers["content-length"] ?? "0") ?? 0
      let body = buffer[end.upperBound...]
      guard body.count >= length else { return nil }
      let json =
        (try? JSONSerialization.jsonObject(with: Data(body.prefix(length)))) as? [String: Any]
      return Received(
        path: requestLine.count > 1 ? String(requestLine[1]) : "",
        headers: headers,
        body: (json ?? [:]).mapValues { SendableJSON($0).value }
      )
    }

    private func respond(to request: Received, on connection: NWConnection) {
      let behavior = lock.withLock {
        received.append(request)
        return self.behavior
      }
      let inputs = request.body["input"] as? [String] ?? []
      let status: Int
      var payload: Any = [:]
      var text: String?
      switch behavior {
      case .vectors(let dimension):
        status = 200
        payload = Self.vectorsPayload(inputs, dimension: dimension)
      case .missingVector(let dimension):
        status = 200
        payload = Self.vectorsPayload(Array(inputs.dropLast()), dimension: dimension)
      case .json(let json):
        status = 200
        payload = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) ?? [:]
      case .text(let code, let body):
        status = code
        text = body
      case .error(let code, let message):
        status = code
        payload = ["error": ["message": message, "type": "invalid_request_error"]]
      }
      let body =
        text.map { Data($0.utf8) } ?? (try? JSONSerialization.data(withJSONObject: payload))
        ?? Data()
      var response = Data(
        """
        HTTP/1.1 \(status) Status\r
        Content-Type: application/json\r
        Content-Length: \(body.count)\r
        Connection: close\r
        \r

        """.utf8
      )
      response.append(body)
      connection.send(content: response, completion: .contentProcessed { _ in connection.cancel() })
    }

    /// Returns a response whose vectors are in reverse order,
    /// to check that the client sorts them by index.
    private static func vectorsPayload(_ inputs: [String], dimension: Int) -> [String: Any] {
      let items = inputs.enumerated().reversed().map { index, input in
        [
          "object": "embedding", "index": index,
          "embedding": vector(for: input, dimension: dimension),
        ]
          as [String: Any]
      }
      return ["object": "list", "data": items, "model": "mock"]
    }

    /// Returns a vector that depends only on the text, with values that aren't unit length.
    static func vector(for text: String, dimension: Int) -> [Double] {
      let seed = text.utf8.reduce(0) { ($0 &* 31 &+ Int($1)) % 1_000 }
      return (0..<dimension).map { Double((seed + $0 * 7) % 13 + 1) }
    }
  }

  /// Converts a JSON value from `JSONSerialization` into a value that is `Sendable`.
  private struct SendableJSON {
    let value: any Sendable

    init(_ value: Any) {
      switch value {
      case let string as String: self.value = string
      case let number as NSNumber: self.value = number.doubleValue
      case let array as [Any]:
        let values = array.map { SendableJSON($0).value }
        self.value =
          values.compactMap { $0 as? String }.count == values.count
          ? values.compactMap { $0 as? String } : values
      case let object as [String: Any]: self.value = object.mapValues { SendableJSON($0).value }
      default: self.value = String(describing: value)
      }
    }
  }

  /// A lock for test state.
  ///
  /// The library has its own lock, which tests don't share.
  final class TestLock<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<Result>(_ body: (inout Value) throws -> Result) rethrows -> Result {
      lock.lock()
      defer { lock.unlock() }
      return try body(&value)
    }
  }
#endif
