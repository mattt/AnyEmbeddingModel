#if canImport(Network)
    import Foundation
    import Testing

    @testable import AnyEmbeddingModel

    #if AsyncHTTPClient
        import AsyncHTTPClient
    #endif

    /// A transport that a test sends requests with.
    struct NamedTransport: Sendable, CustomTestStringConvertible {
        let name: String
        let transport: any HTTPTransport

        var testDescription: String { name }

        /// Every transport that the package provides in this build.
        static var all: [NamedTransport] {
            var transports = [
                NamedTransport(name: "URLSession", transport: URLSession(configuration: .ephemeral)),
                NamedTransport(name: "default", transport: makeDefaultTransport()),
            ]
            #if AsyncHTTPClient
                transports.append(NamedTransport(name: "HTTPClient.shared", transport: HTTPClient.shared))
            #endif
            return transports
        }
    }

    @Suite("Redirects", .serialized)
    struct RedirectTests {
        @Test(
            "A redirect to another origin isn't followed",
            arguments: NamedTransport.all,
            [307, 308]
        )
        func crossOriginRedirect(transport: NamedTransport, status: Int) async throws {
            let target = try MockEmbeddingServer(behavior: .vectors(dimension: 3))
            let targetURL = try await target.start()
            defer { target.stop() }

            for apiKey in ["sk-redirect-test", ""] {
                let location = targetURL.appendingPathComponent("embeddings").absoluteString
                let origin = try MockEmbeddingServer(
                    behavior: .text(status: status, body: "", headers: ["Location": location])
                )
                let originURL = try await origin.start()
                defer { origin.stop() }

                let model = OpenAIEmbeddingModel(
                    baseURL: originURL,
                    apiKey: apiKey,
                    model: "m",
                    dimension: 3,
                    transport: transport.transport
                )
                do {
                    _ = try await model.embed(["a"], for: .document)
                    Issue.record("The embedding succeeded through a redirect to another origin.")
                } catch let error as HTTPTransportError {
                    #expect(error.kind == .httpError)
                    #expect(error.response?.status.code == status)
                }
                #expect(origin.requests.count == 1)
                #expect(target.requests.isEmpty, "The other origin received a request.")
            }
        }
    }
#endif
