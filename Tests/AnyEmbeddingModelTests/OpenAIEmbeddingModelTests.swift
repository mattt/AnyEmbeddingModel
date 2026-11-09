#if canImport(Network)
  import Foundation
  import Testing

  import AnyEmbeddingModel

  @Suite("OpenAIEmbeddingModel", .serialized)
  struct OpenAIEmbeddingModelTests {
    private func withServer(
      _ behavior: MockEmbeddingServer.Behavior,
      _ body: (MockEmbeddingServer, URL) async throws -> Void
    ) async throws {
      let server = try MockEmbeddingServer(behavior: behavior)
      let url = try await server.start()
      defer { server.stop() }
      try await body(server, url)
    }

    @Test("Sends the request format and returns unit vectors in input order")
    func requestAndOrder() async throws {
      try await withServer(.vectors(dimension: 4)) { server, url in
        let model = OpenAIEmbeddingModel(
          baseURL: url,
          model: "mock-model",
          dimension: 4,
          prefixes: .nomic,
          apiKey: "test-key"
        )
        let vectors = try await model.embed(["alpha", "beta", "gamma"], for: .query)

        #expect(vectors.count == 3)
        for (text, vector) in zip(["alpha", "beta", "gamma"], vectors) {
          let raw = MockEmbeddingServer.vector(for: "search_query: " + text, dimension: 4)
          let length = raw.reduce(0) { $0 + $1 * $1 }.squareRoot()
          for (value, expected) in zip(vector, raw.map { $0 / length }) {
            #expect(abs(Double(value) - expected) < 1e-6)
          }
        }

        let request = try #require(server.requests.first)
        #expect(request.path == "/v1/embeddings")
        #expect(request.headers["authorization"] == "Bearer test-key")
        #expect(request.body["model"] as? String == "mock-model")
        #expect(request.body["encoding_format"] as? String == "float")
        #expect(
          request.body["input"] as? [String]
            == ["search_query: alpha", "search_query: beta", "search_query: gamma"])
        #expect(request.body["dimensions"] == nil)
      }
    }

    @Test("Splits a large batch into several requests")
    func batching() async throws {
      try await withServer(.vectors(dimension: 2)) { server, url in
        let model = OpenAIEmbeddingModel(
          baseURL: url, model: "m", dimension: 2, maximumBatchSize: 2)
        let texts = ["a", "b", "c", "d", "e"]
        let vectors = try await model.embed(texts, for: .document)
        #expect(vectors.count == 5)
        #expect(server.requests.map { ($0.body["input"] as? [String])?.count } == [2, 2, 1])
        let single = try await model.embed("c", for: .document)
        #expect(single == vectors[2])
      }
    }

    @Test("Sends the dimension only when asked")
    func dimensions() async throws {
      try await withServer(.vectors(dimension: 3)) { server, url in
        let model = OpenAIEmbeddingModel(
          baseURL: url, model: "m", dimension: 3, requestsDimension: true)
        _ = try await model.embed(["a"], for: .document)
        #expect(server.requests.first?.body["dimensions"] as? Double == 3)
      }
    }

    @Test("Reports the server's error message")
    func serverError() async throws {
      try await withServer(.error(status: 413, message: "Input is too long.")) { _, url in
        let model = OpenAIEmbeddingModel(baseURL: url, model: "m", dimension: 3)
        await #expect(throws: EmbeddingError.httpStatus(413, message: "Input is too long.")) {
          try await model.embed(["a"], for: .document)
        }
      }
    }

    private static let key = "sk-secret/one+two"

    private func model(_ url: URL, apiKey: String? = key) -> OpenAIEmbeddingModel {
      OpenAIEmbeddingModel(baseURL: url, model: "m", dimension: 3, apiKey: apiKey)
    }

    /// Returns the error from one embedding, as all of the text that a log could show.
    private func errorText(_ model: OpenAIEmbeddingModel) async -> (Error?, String) {
      do {
        _ = try await model.embed(["a"], for: .document)
        return (nil, "")
      } catch {
        return (error, "\(error) \(String(reflecting: error)) \(error.localizedDescription)")
      }
    }

    @Test("Removes the key from a JSON error, each time it appears")
    func redactsJSONError() async throws {
      let message = "Bad key \(Self.key). Rejected \(Self.key)."
      try await withServer(.error(status: 401, message: message)) { _, url in
        let (error, text) = await errorText(model(url))
        #expect(
          error as? EmbeddingError
            == .httpStatus(401, message: "Bad key <redacted>. Rejected <redacted>."))
        #expect(!text.contains(Self.key))
      }
    }

    @Test("Removes the key from a plain-text error, each time it appears")
    func redactsPlainTextError() async throws {
      let body = "\(Self.key) is invalid: \(Self.key)"
      try await withServer(.text(status: 403, body: body)) { _, url in
        let (error, text) = await errorText(model(url))
        #expect(
          error as? EmbeddingError == .httpStatus(403, message: "<redacted> is invalid: <redacted>")
        )
        #expect(!text.contains(Self.key))
      }
    }

    @Test(
      "Removes the key in the forms that a server encodes it",
      arguments: [
        #"{"detail": "bad sk-secret\/one+two"}"#, "bad sk%2Dsecret%2Fone%2Btwo",
        "bad sk-secret%2Fone%2Btwo",
      ]
    )
    func redactsEncodedForms(body: String) async throws {
      try await withServer(.text(status: 401, body: body)) { _, url in
        let (_, text) = await errorText(model(url))
        #expect(text.contains("<redacted>"))
        #expect(!text.contains("secret"))
      }
    }

    @Test("Removes the key from an unreadable response")
    func redactsDecodingError() async throws {
      // The response has the wrong type where the decoder expects an array.
      let body = #"{"data": "\#(Self.key)"}"#
      try await withServer(.text(status: 200, body: body)) { _, url in
        let (error, text) = await errorText(model(url))
        #expect(error is EmbeddingError)
        #expect(!text.contains(Self.key))
      }
    }

    @Test("Leaves a message unchanged when there is no key")
    func noKeyNoRedaction() async throws {
      try await withServer(.text(status: 500, body: "sk-secret/one+two broke")) { _, url in
        let (error, _) = await errorText(model(url, apiKey: nil))
        #expect(error as? EmbeddingError == .httpStatus(500, message: "sk-secret/one+two broke"))
      }
    }

    @Test("Printing or dumping a model doesn't show the key")
    func keyIsNotPrinted() {
      let model = model(URL(string: "http://a.example/v1")!)
      var dumped = ""
      dump(model, to: &dumped)
      for text in [String(describing: model), String(reflecting: model), dumped] {
        #expect(!text.contains("secret"))
      }
    }

    @Test("Rejects a vector of the wrong dimension")
    func wrongDimension() async throws {
      try await withServer(.vectors(dimension: 5)) { _, url in
        let model = OpenAIEmbeddingModel(baseURL: url, model: "m", dimension: 3)
        await #expect(throws: EmbeddingError.dimensionMismatch(expected: 3, actual: 5)) {
          try await model.embed(["a"], for: .document)
        }
      }
    }

    @Test("Rejects a response with a missing vector")
    func missingVector() async throws {
      try await withServer(.missingVector(dimension: 3)) { _, url in
        let model = OpenAIEmbeddingModel(baseURL: url, model: "m", dimension: 3)
        await #expect(throws: EmbeddingError.wrongVectorCount(expected: 2, actual: 1)) {
          try await model.embed(["a", "b"], for: .document)
        }
      }
    }

    @Test("Rejects empty text before it sends a request")
    func emptyText() async throws {
      try await withServer(.vectors(dimension: 3)) { server, url in
        let model = OpenAIEmbeddingModel(baseURL: url, model: "m", dimension: 3)
        await #expect(throws: EmbeddingError.emptyText(index: 1)) {
          try await model.embed(["a", "  "], for: .document)
        }
        #expect(server.requests.isEmpty)
      }
    }

    @Test("The identifier depends on settings that change vectors, not on the server")
    func identifier() {
      let a = OpenAIEmbeddingModel(
        baseURL: URL(string: "http://a.example/v1")!, model: "m", dimension: 3)
      let b = OpenAIEmbeddingModel(
        baseURL: URL(string: "http://b.example/v1")!, model: "m", dimension: 3)
      let c = OpenAIEmbeddingModel(
        baseURL: URL(string: "http://a.example/v1")!, model: "m", dimension: 3, prefixes: .e5)
      let d = OpenAIEmbeddingModel(
        baseURL: URL(string: "http://a.example/v1")!, model: "m", dimension: 3, revision: "2")
      #expect(a.identifier == b.identifier)
      #expect(a.identifier != c.identifier)
      #expect(a.identifier != d.identifier)
      #expect(a.identifier.hasPrefix("openai/m/-/3/"))
    }

    @Test("The public initializer can't produce colliding identifiers")
    func identifierCollisions() {
      let url = URL(string: "http://a.example/v1")!
      let a = OpenAIEmbeddingModel(baseURL: url, model: "org/m", dimension: 3, revision: "v/1")
      let b = OpenAIEmbeddingModel(baseURL: url, model: "org/m/v", dimension: 3, revision: "1")
      #expect(a.identifier != b.identifier)
      let c = OpenAIEmbeddingModel(
        baseURL: url,
        model: "m",
        dimension: 3,
        prefixes: EmbeddingPrefixes(query: "a\0", document: "b")
      )
      let d = OpenAIEmbeddingModel(
        baseURL: url,
        model: "m",
        dimension: 3,
        prefixes: EmbeddingPrefixes(query: "a", document: "\0b")
      )
      #expect(c.identifier != d.identifier)
    }
  }
#endif
