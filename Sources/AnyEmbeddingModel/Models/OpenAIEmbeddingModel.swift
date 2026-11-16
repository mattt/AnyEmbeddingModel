import Foundation
import HTTPTypes
#if !os(WASI)
    import HTTPTypesFoundation
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
///   apiKey: "",
///   model: "BAAI/bge-small-en-v1.5",
///   dimension: 384,
///   prefixes: .bge
/// )
/// ```
public struct OpenAIEmbeddingModel: EmbeddingModel {
    /// The base URL of OpenAI's API, `https://api.openai.com/v1/`.
    public static let defaultBaseURL = URL(string: "https://api.openai.com/v1/")!

    /// The URL that the path `embeddings` is added to,
    /// such as `https://api.openai.com/v1/`.
    ///
    /// It ends with a slash,
    /// which the initializer adds if the URL that you pass has none.
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

    /// Returns the API key for a request.
    ///
    /// A closure isn't printed or dumped with the model, so the key isn't either.
    private let tokenProvider: @Sendable () -> String
    private let transport: any HTTPTransport

    /// Creates a model.
    ///
    /// - Parameters:
    ///   - baseURL: The URL that the path `embeddings` is added to.
    ///     The default is ``defaultBaseURL``.
    ///   - apiKey: An expression for the key to send as a bearer token,
    ///     such as a string literal or a call like `currentAPIKey()`.
    ///     The parameter is an autoclosure,
    ///     so the model evaluates the expression again, once for each request,
    ///     and an expression that calls a function can return a new key after a rotation.
    ///     An empty key sends no `Authorization` header.
    ///     If a response contains the key that its request sent,
    ///     errors omit the server's text and say that it was withheld.
    ///     The model looks for the key after it decodes JSON escapes and percent-encoding,
    ///     in any ASCII case, and it also withholds text that it can't decode with confidence.
    ///     It can't detect part of a key, such as a masked copy,
    ///     or a key in another encoding, such as Base64.
    ///   - model: The name of the model on the server.
    ///   - dimension: The number of elements in each vector.
    ///   - requestsDimension: Whether to send `dimension` in each request.
    ///   - prefixes: The text before each query and document.
    ///   - revision: A string that changes ``identifier``
    ///     when the server changes the model behind the same name.
    ///   - maximumBatchSize: The maximum number of texts in one request.
    ///   - transport: How the model sends its requests: by default ``makeDefaultTransport()``.
    public init(
        baseURL: URL = defaultBaseURL,
        apiKey tokenProvider: @escaping @autoclosure @Sendable () -> String,
        model: String,
        dimension: Int,
        requestsDimension: Bool = false,
        prefixes: EmbeddingPrefixes = .none,
        revision: String? = nil,
        maximumBatchSize: Int = 64,
        transport: any HTTPTransport = makeDefaultTransport()
    ) {
        precondition(dimension > 0, "The dimension must be greater than zero.")
        precondition(maximumBatchSize > 0, "The batch size must be greater than zero.")
        var baseURL = baseURL
        if !baseURL.path.hasSuffix("/") {
            baseURL = baseURL.appendingPathComponent("")
        }
        self.baseURL = baseURL
        self.model = model
        self.dimension = dimension
        self.requestsDimension = requestsDimension
        self.prefixes = prefixes
        self.revision = revision
        self.tokenProvider = tokenProvider
        self.maximumBatchSize = maximumBatchSize
        self.transport = transport
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
    /// - Throws: ``HTTPTransportError`` if the server rejects a request,
    ///   for example because a text is too long,
    ///   or if the model can't read the response;
    ///   another ``EmbeddingError``;
    ///   or the transport's error if a request fails on the way or is cancelled.
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

    /// The message in an error when the server's own text might contain the API key.
    static let withheldMessage =
        "The server's response was withheld because it might contain the API key."

    /// Returns `error`, or an error without the server's text
    /// if anything that `error` keeps might disclose the API key.
    ///
    /// A server can echo the credential that it rejected,
    /// and text from a server reaches errors, and errors reach logs.
    /// The check covers the whole body,
    /// the status's reason phrase, every header,
    /// and the description of the underlying error,
    /// so a key anywhere in the response withholds all of the server's text.
    /// The error that replaces it keeps the kind and the status code,
    /// and its body is ``withheldMessage``.
    /// ``CredentialScreen`` describes what it can't detect.
    static func screened(_ error: HTTPTransportError, with screen: CredentialScreen?) -> HTTPTransportError {
        guard let screen else { return error }
        var texts: [[UInt8]] = []
        if let body = error.body { texts.append(Array(body)) }
        if let response = error.response {
            texts.append(Array(response.status.reasonPhrase.utf8))
            for field in response.headerFields {
                texts.append(Array(field.name.rawName.utf8))
                texts.append(Array(field.value.utf8))
            }
        }
        if let underlying = error.underlyingError {
            texts.append(Array(String(describing: underlying).utf8))
            texts.append(Array(String(reflecting: underlying).utf8))
        }
        guard !texts.allSatisfy(screen.isClear) else { return error }
        return HTTPTransportError(
            kind: error.kind,
            response: error.response.map { HTTPResponse(status: .init(code: $0.status.code)) },
            body: Data(withheldMessage.utf8)
        )
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

    /// Sends one request and returns the vectors in the order of `inputs`.
    private func request(_ inputs: [String]) async throws -> [[Float]] {
        var fields = HTTPFields()
        // The key is evaluated once,
        // so the response is screened for the key that this request sent.
        let apiKey = tokenProvider()
        if !apiKey.isEmpty {
            fields[.authorization] = "Bearer \(apiKey)"
        }
        let request = HTTPRequest(
            method: .post,
            url: baseURL.appendingPathComponent("embeddings"),
            headerFields: fields
        )
        let body = try JSONEncoder().encode(
            Request(model: model, input: inputs, dimensions: requestsDimension ? dimension : nil)
        )
        let screen = apiKey.isEmpty ? nil : CredentialScreen(credential: apiKey)

        let data: Data
        do {
            data = try await transport.fetch(request, body: body)
        } catch let error as HTTPTransportError {
            throw Self.screened(error, with: screen)
        }
        let decoded: Response
        do {
            decoded = try JSONDecoder().decode(Response.self, from: data)
        } catch {
            // A decoding error can quote a value from the response.
            throw Self.screened(
                HTTPTransportError(kind: .decodingError, body: data, underlyingError: error),
                with: screen
            )
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
