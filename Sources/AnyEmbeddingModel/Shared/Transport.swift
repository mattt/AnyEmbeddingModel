import Foundation
import HTTPTypes

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif
#if AsyncHTTPClient
    import AsyncHTTPClient
#endif

/// How a model sends its HTTP requests.
///
/// ``OpenAIEmbeddingModel`` sends every request through one, given as its `transport`,
/// and reads its response in ``send(request:body:timeout:responseHandler:)``.
/// `URLSession` is one,
/// and so is AsyncHTTPClient's `HTTPClient` with the `AsyncHTTPClient` trait.
///
/// Supply your own to decide how requests are sent,
/// such as one that waits for a free slot of an account that takes few requests at once,
/// or one that observes them, then passes each on to another transport.
///
/// This protocol has the same shape as AnyLanguageModel's `HTTPTransport`.
public protocol HTTPTransport: Sendable {
    /// Sends `request` with `body`, and passes its response to `responseHandler`.
    ///
    /// A transport keeps to this contract:
    /// - It calls `responseHandler` at most once, with the response to `request`.
    /// - Cancelling the calling task cancels the request, and `send` throws:
    ///   a body cut short by cancelling throws as it's read,
    ///   rather than ending as if it were whole.
    /// - The response's body can be read only in `responseHandler`:
    ///   once it returns, the request is over, and the body can't be read anymore.
    /// - A transport that retries a request doesn't retry it
    ///   once it has passed a response to `responseHandler`.
    /// - It doesn't follow a redirect that changes the scheme, the host, or the port.
    ///   It passes the redirect response to `responseHandler` instead,
    ///   or throws without calling it.
    ///
    /// The package's `URLSession` transport keeps every part of this contract.
    /// AsyncHTTPClient's `HTTPClient` follows redirects as its own configuration says,
    /// and the transport can't change that for a client that you supply.
    /// Such a client never resends the body to another origin,
    /// but it can follow a redirect that drops the body, such as a 303,
    /// to another origin as a `GET` with no body and no `Authorization` header;
    /// the transport then throws rather than pass that response on.
    /// The client that ``makeDefaultTransport()`` returns follows no redirects,
    /// so it keeps the whole contract.
    ///
    /// - Parameters:
    ///   - request: The request to send.
    ///   - body: What the request sends, if anything.
    ///   - timeout: How long to wait for the response to start,
    ///     or `nil` when the caller didn't say, for the transport to decide.
    ///     The transport decides when that time starts,
    ///     so a request that waits for a free slot before it's sent doesn't use up its deadline.
    ///     AsyncHTTPClient applies it only until the response starts,
    ///     and waits 180 seconds without one.
    ///     A URL session applies it to every pause in the response too,
    ///     and without one waits as long as its own `timeoutIntervalForRequest`.
    ///   - responseHandler: Reads the response, its body as it arrives.
    func send<Return>(
        request: HTTPRequest,
        body: Data?,
        timeout: Duration?,
        responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
    ) async throws -> Return
}

/// The body of an HTTP response, in chunks of data as they arrive.
///
/// It's read once, in the response handler it's given to.
public struct HTTPResponseBody: AsyncSequence {
    public typealias Element = Data

    private let source: ChunkSource

    /// A body read from `chunks`.
    public init<Chunks: AsyncSequence>(_ chunks: Chunks) where Chunks.Element == Data {
        source = Iterating(chunks.makeAsyncIterator())
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(source: source)
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        fileprivate let source: ChunkSource

        public mutating func next() async throws -> Data? {
            try await source.next()
        }
    }

    /// The whole body, as data.
    public func collect() async throws -> Data {
        var data = Data()
        for try await chunk in self { data.append(chunk) }
        return data
    }
}

/// Where a body's chunks come from, whatever sequence gives them.
///
/// A base class with a generic subclass hides the sequence's type,
/// so a body is one type whatever the transport reads it from.
/// `any AsyncIteratorProtocol<Data, any Error>` would say the same directly,
/// but it needs macOS 15 and the package supports macOS 14:
/// once it doesn't, this and ``Iterating`` become that existential.
private class ChunkSource {
    func next() async throws -> Data? { nil }
}

private final class Iterating<Iterator: AsyncIteratorProtocol>: ChunkSource where Iterator.Element == Data {
    private var iterator: Iterator

    init(_ iterator: Iterator) {
        self.iterator = iterator
    }

    override func next() async throws -> Data? {
        try await iterator.next()
    }
}

/// Returns whether two URLs have the same scheme, host, and port,
/// counting a missing port as the scheme's default port.
func haveSameOrigin(_ first: URL, _ second: URL) -> Bool {
    func origin(_ url: URL) -> (scheme: String?, host: String?, port: Int?) {
        let scheme = url.scheme?.lowercased()
        let defaultPort: Int? =
            switch scheme {
            case "http": 80
            case "https": 443
            default: nil
            }
        return (scheme, url.host?.lowercased(), url.port ?? defaultPort)
    }
    let a = origin(first)
    let b = origin(second)
    return a.scheme != nil && a.host != nil && a.scheme == b.scheme && a.host == b.host && a.port == b.port
}

#if AsyncHTTPClient
    /// The transport a remote model sends its requests with unless it's given another:
    /// an AsyncHTTPClient client with the shared client's configuration,
    /// except that it follows no redirects.
    public func makeDefaultTransport() -> any HTTPTransport {
        defaultHTTPClient
    }

    /// A client like `HTTPClient.shared` that doesn't follow redirects.
    ///
    /// It's never shut down, as it lasts as long as the process.
    private let defaultHTTPClient: HTTPClient = {
        var configuration = HTTPClient.Configuration.singletonConfiguration
        configuration.redirectConfiguration = .disallow
        return HTTPClient(eventLoopGroup: HTTPClient.defaultEventLoopGroup, configuration: configuration)
    }()
#elseif os(WASI)
    /// WASI has no networking of its own, so a remote model needs its `transport` given,
    /// such as one that sends requests with the host's `fetch`.
    public func makeDefaultTransport() -> any HTTPTransport {
        MissingTransport()
    }

    private struct MissingTransport: HTTPTransport {
        func send<Return>(
            request: HTTPRequest,
            body: Data?,
            timeout: Duration?,
            responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
        ) async throws -> Return {
            throw HTTPTransportError(kind: .invalidRequest, underlyingError: WASITransportError())
        }
    }

    private struct WASITransportError: Error, CustomStringConvertible {
        var description: String { "WASI has no default transport: give the model a `transport`." }
    }
#else
    /// The transport a remote model sends its requests with unless it's given another:
    /// a URL session with the default configuration.
    public func makeDefaultTransport() -> any HTTPTransport {
        URLSession(configuration: .default)
    }
#endif
