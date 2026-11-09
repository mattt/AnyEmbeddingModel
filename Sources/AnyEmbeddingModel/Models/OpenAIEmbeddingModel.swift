import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// An embedding model on a server that accepts the OpenAI embeddings request format.
///
/// The model sends `POST {baseURL}/embeddings`
/// with a JSON body of the form
/// `{"model": "...", "input": ["..."], "encoding_format": "float"}`
/// and reads the vectors from the `data` array of the response.
/// OpenAI, Hugging Face Text Embeddings Inference,
/// Ollama, llama.cpp's server, LM Studio, and vLLM accept this format.
///
/// The format has no field for the purpose of a text,
/// so the model adds ``prefixes`` to each text before it sends the request.
///
/// ```swift
/// let model = OpenAIEmbeddingModel(
///   baseURL: URL(string: "http://localhost:8080/v1")!,
///   model: "BAAI/bge-small-en-v1.5",
///   dimension: 384,
///   prefixes: .bge
/// )
/// ```
public struct OpenAIEmbeddingModel: EmbeddingModel {
  /// The URL that the path `embeddings` is added to,
  /// such as `https://api.openai.com/v1`.
  public let baseURL: URL

  /// The name of the model on the server.
  public let model: String

  /// The number of elements in each vector.
  ///
  /// The model checks each vector from the server against this value.
  public let dimension: Int

  /// Whether the request asks the server for ``dimension`` elements.
  ///
  /// Some models, such as OpenAI's `text-embedding-3` models,
  /// can return shorter vectors.
  /// Servers that can't return them may reject the field.
  public let requestsDimension: Bool

  /// The text that the model puts before each input.
  public let prefixes: EmbeddingPrefixes

  /// A string that you change when the server changes the model behind the same name.
  public let revision: String?

  /// The maximum number of texts in one request.
  ///
  /// The model splits a larger batch into several requests, one after another.
  public let maximumBatchSize: Int

  /// A string that identifies the vectors that this model produces.
  ///
  /// The identifier has the model name, ``revision``, ``dimension``,
  /// and a fingerprint of ``prefixes`` and ``requestsDimension``.
  /// It doesn't have the server's URL,
  /// and it can't detect that a server changed the model behind a name.
  public let identifier: String

  private let apiKey: APIKey?
  private let session: URLSession

  /// Creates a model.
  ///
  /// - Parameters:
  ///   - baseURL: The URL that the path `embeddings` is added to.
  ///   - model: The name of the model on the server.
  ///   - dimension: The number of elements in each vector.
  ///   - requestsDimension: Whether to send `dimension` in each request.
  ///   - prefixes: The text before each query and document.
  ///   - revision: A string that changes ``identifier``
  ///     when the server changes the model behind the same name.
  ///   - apiKey: A key to send as a bearer token, or `nil` to send none.
  ///   - maximumBatchSize: The maximum number of texts in one request.
  ///   - session: The URL session that sends the requests.
  public init(
    baseURL: URL,
    model: String,
    dimension: Int,
    requestsDimension: Bool = false,
    prefixes: EmbeddingPrefixes = .none,
    revision: String? = nil,
    apiKey: String? = nil,
    maximumBatchSize: Int = 64,
    session: URLSession = .shared
  ) {
    precondition(dimension > 0, "The dimension must be greater than zero.")
    precondition(maximumBatchSize > 0, "The batch size must be greater than zero.")
    self.baseURL = baseURL
    self.model = model
    self.dimension = dimension
    self.requestsDimension = requestsDimension
    self.prefixes = prefixes
    self.revision = revision
    self.apiKey = apiKey.map(APIKey.init)
    self.maximumBatchSize = maximumBatchSize
    self.session = session
    self.identifier = Identifier.make(
      engine: "openai",
      model: model,
      revision: revision,
      dimension: dimension,
      settings: [prefixes.query, prefixes.document, requestsDimension ? "dimensions" : ""]
    )
  }

  /// Returns a vector for each text.
  ///
  /// A batch larger than ``maximumBatchSize`` becomes several requests, one after another.
  ///
  /// - Throws: ``EmbeddingError/httpStatus(_:message:)`` if the server rejects a request,
  ///   for example because a text is too long,
  ///   or another ``EmbeddingError`` or a `URLError`.
  public func embed(_ texts: [String], for purpose: EmbeddingPurpose) async throws -> [[Float]] {
    try Vectors.validate(texts)
    var vectors: [[Float]] = []
    vectors.reserveCapacity(texts.count)
    for batch in Vectors.batches(texts, size: maximumBatchSize) {
      try Task.checkCancellation()
      let inputs = batch.map { prefixes.apply(to: $0, for: purpose) }
      let batchVectors = try await request(inputs)
      for (offset, vector) in batchVectors.enumerated() {
        vectors.append(
          try Vectors.normalized(vector, dimension: dimension, index: batch.startIndex + offset)
        )
      }
    }
    return vectors
  }

  /// Returns `text` with every occurrence of the API key replaced.
  ///
  /// A server can echo the credential that it rejected.
  /// Text from a server reaches errors, and errors reach logs.
  private func redacted(_ text: String) -> String {
    guard let apiKey else { return text }
    return apiKey.redact(from: text)
  }

  /// An API key that doesn't appear when the model is printed or dumped.
  private struct APIKey: Sendable, CustomStringConvertible, CustomDebugStringConvertible,
    CustomReflectable
  {
    let value: String

    var description: String { "<redacted>" }
    var debugDescription: String { "<redacted>" }
    var customMirror: Mirror { Mirror(self, children: [:]) }

    /// Returns `text` with every form of the key that a server could send replaced:
    /// the key itself, as it appears escaped in JSON, and percent-encoded.
    func redact(from text: String) -> String {
      guard !value.isEmpty else { return text }
      let escaped =
        (try? JSONEncoder().encode(value))
        .map { String(decoding: $0.dropFirst().dropLast(), as: UTF8.self) } ?? value
      let forms = [
        value,
        escaped,
        escaped.replacing("/", with: "\\/"),
        value.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? value,
        value.addingPercentEncoding(
          withAllowedCharacters: CharacterSet.alphanumerics.union(
            CharacterSet(charactersIn: "-._~"))
        ) ?? value,
      ]
      // Longer forms first, so that a shorter form can't split a longer one.
      return Set(forms).sorted { $0.count > $1.count }.reduce(text) {
        $0.replacing($1, with: "<redacted>")
      }
    }
  }

  private struct Request: Encodable {
    var model: String
    var input: [String]
    var encodingFormat = "float"
    var dimensions: Int?

    enum CodingKeys: String, CodingKey {
      case model, input, dimensions
      case encodingFormat = "encoding_format"
    }
  }

  private struct Response: Decodable {
    struct Item: Decodable {
      var index: Int
      var embedding: [Float]
    }
    var data: [Item]
  }

  private struct ErrorResponse: Decodable {
    struct Detail: Decodable { var message: String }
    var error: Detail
  }

  /// Sends one request and returns the vectors in the order of `inputs`.
  private func request(_ inputs: [String]) async throws -> [[Float]] {
    var request = URLRequest(url: baseURL.appendingPathComponent("embeddings"))
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let apiKey {
      request.setValue("Bearer \(apiKey.value)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONEncoder().encode(
      Request(model: model, input: inputs, dimensions: requestsDimension ? dimension : nil)
    )

    let (data, response) = try await session.data(for: request)
    guard let http = response as? HTTPURLResponse else {
      throw EmbeddingError.invalidResponse("The response isn't an HTTP response.")
    }
    guard (200..<300).contains(http.statusCode) else {
      let message =
        (try? JSONDecoder().decode(ErrorResponse.self, from: data).error.message)
        ?? String(decoding: data, as: UTF8.self)
      throw EmbeddingError.httpStatus(http.statusCode, message: redacted(message))
    }

    let decoded: Response
    do {
      decoded = try JSONDecoder().decode(Response.self, from: data)
    } catch {
      // A decoding error can quote a value from the response.
      throw EmbeddingError.invalidResponse(redacted(String(describing: error)))
    }
    guard decoded.data.count == inputs.count else {
      throw EmbeddingError.wrongVectorCount(expected: inputs.count, actual: decoded.data.count)
    }
    // The format gives each vector an index,
    // and doesn't promise that the array is in order.
    var vectors = [[Float]?](repeating: nil, count: inputs.count)
    for item in decoded.data {
      guard vectors.indices.contains(item.index), vectors[item.index] == nil else {
        throw EmbeddingError.invalidResponse("The response has a bad index: \(item.index).")
      }
      vectors[item.index] = item.embedding
    }
    // Every index is in range and unique, and there is one item for each input,
    // so every element is set.
    return vectors.compactMap { $0 }
  }
}
