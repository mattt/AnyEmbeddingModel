#if canImport(Network)
    import Foundation
    import Testing

    @testable import AnyEmbeddingModel

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
                    apiKey: "test-key",
                    model: "mock-model",
                    dimension: 4,
                    prefixes: .nomic
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
                        == ["search_query: alpha", "search_query: beta", "search_query: gamma"]
                )
                #expect(request.body["dimensions"] == nil)
            }
        }

        @Test("Splits a large batch into several requests")
        func batching() async throws {
            try await withServer(.vectors(dimension: 2)) { server, url in
                let model = OpenAIEmbeddingModel(
                    baseURL: url,
                    apiKey: "",
                    model: "m",
                    dimension: 2,
                    maximumBatchSize: 2
                )
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
                    baseURL: url,
                    apiKey: "",
                    model: "m",
                    dimension: 3,
                    requestsDimension: true
                )
                _ = try await model.embed(["a"], for: .document)
                #expect(server.requests.first?.body["dimensions"] as? Double == 3)
            }
        }

        /// The parts of an ``HTTPTransportError`` that the tests compare.
        private struct Failure: Equatable {
            var kind: HTTPTransportError.Kind
            var status: Int?
            var body: String?
        }

        /// Returns the parts of `error` that the tests compare,
        /// or `nil` if it isn't an ``HTTPTransportError``.
        private func transportFailure(_ error: (any Error)?) -> Failure? {
            guard let error = error as? HTTPTransportError else { return nil }
            return Failure(
                kind: error.kind,
                status: error.response?.status.code,
                body: error.body.map { String(decoding: $0, as: UTF8.self) }
            )
        }

        @Test("Reports the server's error status and body")
        func serverError() async throws {
            try await withServer(.error(status: 413, message: "Input is too long.")) { _, url in
                let model = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "m", dimension: 3)
                let (error, _) = await errorText(model)
                // A local named like a function that a Testing macro calls
                // shadows that function in Swift 6.3's expansion.
                let reported = try #require(transportFailure(error))
                #expect(reported.kind == .httpError)
                #expect(reported.status == 413)
                #expect(reported.body?.contains("Input is too long.") == true)
            }
        }

        private static let key = "sk-secret/one+two"

        private func model(_ url: URL, apiKey: String = key) -> OpenAIEmbeddingModel {
            OpenAIEmbeddingModel(baseURL: url, apiKey: apiKey, model: "m", dimension: 3)
        }

        /// Returns the error from one embedding, as all of the text that a log could show.
        private func errorText(_ model: OpenAIEmbeddingModel) async -> (Error?, String) {
            do {
                _ = try await model.embed(["a"], for: .document)
                return (nil, "")
            } catch {
                var dumped = ""
                dump(error, to: &dumped)
                return (error, "\(error) \(String(reflecting: error)) \(error.localizedDescription) \(dumped)")
            }
        }

        private static let withheld = OpenAIEmbeddingModel.withheldMessage

        @Test("Withholds a JSON error that contains the key")
        func withholdsJSONError() async throws {
            let message = "Bad key \(Self.key). Rejected \(Self.key)."
            try await withServer(.error(status: 401, message: message)) { _, url in
                let (error, text) = await errorText(model(url))
                #expect(transportFailure(error) == Failure(kind: .httpError, status: 401, body: Self.withheld))
                #expect(!text.contains(Self.key))
            }
        }

        @Test("Withholds a plain-text error that contains the key")
        func withholdsPlainTextError() async throws {
            let body = "\(Self.key) is invalid: \(Self.key)"
            try await withServer(.text(status: 403, body: body)) { _, url in
                let (error, text) = await errorText(model(url))
                #expect(transportFailure(error) == Failure(kind: .httpError, status: 403, body: Self.withheld))
                #expect(!text.contains(Self.key))
            }
        }

        @Test(
            "Withholds the key in each form that a server can encode it",
            arguments: [
                #"{"detail":"sk-secret/one+two"}"#,
                #"{"detail": "bad sk-secret\/one+two"}"#,
                #"{"detail":"\u0073k-secret/one+two"}"#,
                #"{"detail":"\u0073\u006B-secret\u002Fone\u002btwo"}"#,
                "bad sk%2Dsecret%2Fone%2Btwo",
                "bad sk-secret%2Fone%2Btwo",
                "sk-secret%2fone%2btwo",
                "sk-secret%252Fone%252Btwo",
                "SK-SECRET/ONE+TWO",
                #"{"error":{"message":"bad","param":"sk-secret/one+two"}}"#,
            ]
        )
        func withholdsEncodedForms(body: String) async throws {
            try await withServer(.text(status: 401, body: body)) { _, url in
                let (error, text) = await errorText(model(url))
                #expect(transportFailure(error) == Failure(kind: .httpError, status: 401, body: Self.withheld))
                #expect(!text.lowercased().contains("secret/one"))
                #expect(!text.contains("%2"))
            }
        }

        @Test("Withholds an unreadable response that contains the key")
        func withholdsDecodingError() async throws {
            // The response has the wrong type where the decoder expects an array.
            for value in [Self.key, "sk-secret%2fone%2btwo"] {
                let body = #"{"data": "\#(value)"}"#
                try await withServer(.text(status: 200, body: body)) { _, url in
                    let (error, text) = await errorText(model(url))
                    #expect(transportFailure(error) == Failure(kind: .decodingError, body: Self.withheld))
                    #expect((error as? HTTPTransportError)?.underlyingError == nil)
                    #expect(!text.contains(value))
                }
            }
        }

        @Test("Keeps the server's text when it doesn't contain the key")
        func keepsOrdinaryErrors() async throws {
            try await withServer(.text(status: 503, body: "Overloaded: 100% of capacity")) { _, url in
                let (error, _) = await errorText(model(url))
                #expect(
                    transportFailure(error)
                        == Failure(kind: .httpError, status: 503, body: "Overloaded: 100% of capacity")
                )
            }
            try await withServer(.text(status: 200, body: #"{"data": "nope"}"#)) { _, url in
                let (error, _) = await errorText(model(url))
                #expect(transportFailure(error) == Failure(kind: .decodingError, body: #"{"data": "nope"}"#))
                #expect((error as? HTTPTransportError)?.underlyingError is DecodingError)
            }
        }

        @Test("Withholds the server's text when a header contains the key")
        func withholdsHeader() async throws {
            let headers = ["WWW-Authenticate": #"Bearer error="invalid", key="sk-secret/one+two""#]
            try await withServer(.text(status: 401, body: "denied", headers: headers)) { _, url in
                let (error, text) = await errorText(model(url))
                #expect(transportFailure(error) == Failure(kind: .httpError, status: 401, body: Self.withheld))
                #expect((error as? HTTPTransportError)?.response?.headerFields.isEmpty == true)
                #expect(!text.contains(Self.key))
            }
        }

        @Test("Leaves a message unchanged when there is no key")
        func noKeyNoRedaction() async throws {
            try await withServer(.text(status: 500, body: "sk-secret/one+two broke")) { _, url in
                let (error, _) = await errorText(model(url, apiKey: ""))
                #expect(
                    transportFailure(error) == Failure(kind: .httpError, status: 500, body: "sk-secret/one+two broke")
                )
            }
        }

        @Test("The base URL defaults to OpenAI's and always ends with a slash")
        func baseURL() async throws {
            #expect(OpenAIEmbeddingModel.defaultBaseURL.absoluteString == "https://api.openai.com/v1/")
            #expect(
                OpenAIEmbeddingModel(apiKey: "", model: "m", dimension: 3).baseURL
                    == OpenAIEmbeddingModel.defaultBaseURL
            )
            for path in ["v1", "v1/"] {
                let model = OpenAIEmbeddingModel(
                    baseURL: URL(string: "http://a.example/\(path)")!,
                    apiKey: "",
                    model: "m",
                    dimension: 3
                )
                #expect(model.baseURL.absoluteString == "http://a.example/v1/")
            }
            try await withServer(.vectors(dimension: 3)) { server, url in
                let slashed = url.appendingPathComponent("")
                #expect(slashed.absoluteString.hasSuffix("/v1/"))
                let model = OpenAIEmbeddingModel(baseURL: slashed, apiKey: "", model: "m", dimension: 3)
                _ = try await model.embed(["a"], for: .document)
                #expect(server.requests.map(\.path) == ["/v1/embeddings"])
            }
        }

        @Test("An empty key sends no Authorization header")
        func emptyKey() async throws {
            try await withServer(.vectors(dimension: 3)) { server, url in
                let model = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "m", dimension: 3)
                _ = try await model.embed(["a"], for: .document)
                let request = try #require(server.requests.first)
                #expect(request.headers["authorization"] == nil)
            }
        }

        @Test("The key is evaluated once for each request")
        func keyPerRequest() async throws {
            let calls = TestLock(0)
            let key: @Sendable () -> String = {
                calls.withLock { count in
                    count += 1
                    return "sk-\(count)"
                }
            }
            try await withServer(.vectors(dimension: 3)) { server, url in
                let model = OpenAIEmbeddingModel(
                    baseURL: url,
                    apiKey: key(),
                    model: "m",
                    dimension: 3,
                    maximumBatchSize: 2
                )
                _ = try await model.embed(["a", "b", "c", "d", "e"], for: .document)
                #expect(calls.withLock { $0 } == 3)
                #expect(
                    server.requests.map { $0.headers["authorization"] } == [
                        "Bearer sk-1", "Bearer sk-2", "Bearer sk-3",
                    ]
                )
            }
        }

        @Test("A response is screened for the key that its request sent, after the key rotates")
        func rotatedKey() async throws {
            let calls = TestLock(0)
            let key: @Sendable () -> String = {
                calls.withLock { count in
                    count += 1
                    return count == 1 ? "sk-first-key" : "sk-second-key"
                }
            }
            try await withServer(.echoAuthorization(status: 401)) { _, url in
                let model = OpenAIEmbeddingModel(baseURL: url, apiKey: key(), model: "m", dimension: 3)
                let (first, firstText) = await errorText(model)
                #expect(transportFailure(first) == Failure(kind: .httpError, status: 401, body: Self.withheld))
                #expect(!firstText.contains("sk-first-key"))
                let (second, secondText) = await errorText(model)
                #expect(transportFailure(second) == Failure(kind: .httpError, status: 401, body: Self.withheld))
                #expect(!secondText.contains("sk-second-key"))
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
                let model = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "m", dimension: 3)
                await #expect(throws: EmbeddingError.dimensionMismatch(expected: 3, actual: 5)) {
                    try await model.embed(["a"], for: .document)
                }
            }
        }

        @Test("Rejects a response with a missing vector")
        func missingVector() async throws {
            try await withServer(.missingVector(dimension: 3)) { _, url in
                let model = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "m", dimension: 3)
                await #expect(throws: EmbeddingError.wrongVectorCount(expected: 2, actual: 1)) {
                    try await model.embed(["a", "b"], for: .document)
                }
            }
        }

        @Test("Rejects empty text before it sends a request")
        func emptyText() async throws {
            try await withServer(.vectors(dimension: 3)) { server, url in
                let model = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "m", dimension: 3)
                await #expect(throws: EmbeddingError.emptyText(index: 1)) {
                    try await model.embed(["a", "  "], for: .document)
                }
                #expect(server.requests.isEmpty)
            }
        }

        @Test("The identifier depends on settings that change vectors, not on the server")
        func identifier() {
            let a = OpenAIEmbeddingModel(
                baseURL: URL(string: "http://a.example/v1")!,
                apiKey: "",
                model: "m",
                dimension: 3
            )
            let b = OpenAIEmbeddingModel(
                baseURL: URL(string: "http://b.example/v1")!,
                apiKey: "",
                model: "m",
                dimension: 3
            )
            let c = OpenAIEmbeddingModel(
                baseURL: URL(string: "http://a.example/v1")!,
                apiKey: "",
                model: "m",
                dimension: 3,
                prefixes: .e5
            )
            let d = OpenAIEmbeddingModel(
                baseURL: URL(string: "http://a.example/v1")!,
                apiKey: "",
                model: "m",
                dimension: 3,
                revision: "2"
            )
            #expect(a.identifier == b.identifier)
            #expect(a.identifier != c.identifier)
            #expect(a.identifier != d.identifier)
            #expect(a.identifier.hasPrefix("openai/m/-/3/"))
        }

        /// The expected values come from `mise run generate:goldens`.
        @Test("Identifiers match the independent implementation")
        func identifierGoldens() {
            let url = URL(string: "http://a.example/v1")!
            #expect(
                OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "text-embedding-3-small", dimension: 1536)
                    .identifier == "openai/text-embedding-3-small/-/1536/11dac35ef0813626"
            )
            #expect(
                OpenAIEmbeddingModel(
                    baseURL: url,
                    apiKey: "",
                    model: "nomic-embed-text:latest",
                    dimension: 768,
                    requestsDimension: true,
                    prefixes: .nomic,
                    revision: "2"
                ).identifier == "openai/nomic-embed-text%3Alatest/2/768/99860da7f4726551"
            )
            #expect(
                OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "mod\u{E8}le", dimension: 3).identifier
                    == "openai/mod%C3%A8le/-/3/11dac35ef0813626"
            )
            #expect(
                OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "mode\u{300}le", dimension: 3).identifier
                    == "openai/mode%CC%80le/-/3/11dac35ef0813626"
            )
        }

        @Test("The public initializer can't produce colliding identifiers")
        func identifierCollisions() {
            let url = URL(string: "http://a.example/v1")!
            let a = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "org/m", dimension: 3, revision: "v/1")
            let b = OpenAIEmbeddingModel(baseURL: url, apiKey: "", model: "org/m/v", dimension: 3, revision: "1")
            #expect(a.identifier != b.identifier)
            let c = OpenAIEmbeddingModel(
                baseURL: url,
                apiKey: "",
                model: "m",
                dimension: 3,
                prefixes: EmbeddingPrefixes(query: "a\0", document: "b")
            )
            let d = OpenAIEmbeddingModel(
                baseURL: url,
                apiKey: "",
                model: "m",
                dimension: 3,
                prefixes: EmbeddingPrefixes(query: "a", document: "\0b")
            )
            #expect(c.identifier != d.identifier)
        }
    }
#endif
