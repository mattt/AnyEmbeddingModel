#if AsyncHTTPClient
    import AsyncHTTPClient
    import Foundation
    import HTTPTypes
    import NIOCore
    import NIOHTTP1

    /// AsyncHTTPClient sends a model's requests, reading each response's body as it arrives.
    ///
    /// A client follows redirects as its configuration says,
    /// and a request can't change that,
    /// so this transport keeps redirects within the request's origin in two ways.
    /// It sends the body as a sequence that can be read only once,
    /// and a client doesn't follow a redirect that would send the body again, such as a 307 or 308:
    /// it returns that redirect response instead.
    /// A client can still follow a redirect that drops the body,
    /// such as a 303, which makes the request a `GET`;
    /// when the origin changes, AsyncHTTPClient removes the `Authorization` header from it.
    /// If a response comes from another origin after such a redirect,
    /// this transport throws ``HTTPTransportError`` with ``HTTPTransportError/Kind/invalidResponse``
    /// rather than pass the response on.
    /// ``makeDefaultTransport()`` returns a client that follows no redirects.
    extension HTTPClient: HTTPTransport {
        public func send<Return>(
            request: HTTPRequest,
            body: Data?,
            timeout: Duration?,
            responseHandler: (HTTPResponse, HTTPResponseBody) async throws -> Return
        ) async throws -> Return {
            guard let scheme = request.scheme, let authority = request.authority, let path = request.path,
                let url = URL(string: "\(scheme)://\(authority)\(path)")
            else {
                throw HTTPTransportError(kind: .invalidRequest)
            }
            var sent = HTTPClientRequest(url: url.absoluteString)
            sent.method = HTTPMethod(rawValue: request.method.rawValue)
            for field in request.headerFields {
                sent.headers.add(name: field.name.rawName, value: field.value)
            }
            if let body {
                // A stream can't be sent twice, so a client returns a redirect that would resend it.
                let buffer = ByteBuffer(bytes: body)
                let chunks = AsyncStream<ByteBuffer> { continuation in
                    continuation.yield(buffer)
                    continuation.finish()
                }
                sent.body = .stream(chunks, length: .known(Int64(body.count)))
            }

            // A request waits at most 180 seconds for its response to start unless it says otherwise.
            // In nanoseconds, as far as they go:
            // a timeout too long to count in them has no limit.
            let (seconds, attoseconds) = (timeout ?? .seconds(180)).components
            let (wholeNanoseconds, multiplicationOverflowed) = seconds.multipliedReportingOverflow(by: 1_000_000_000)
            let (nanoseconds, additionOverflowed) = wholeNanoseconds.addingReportingOverflow(
                attoseconds / 1_000_000_000
            )
            let overflowed = multiplicationOverflowed || additionOverflowed
            let received = try await execute(sent, timeout: .nanoseconds(overflowed ? .max : nanoseconds))
            for hop in received.history {
                guard let hopURL = URL(string: hop.request.url), haveSameOrigin(url, hopURL) else {
                    throw HTTPTransportError(kind: .invalidResponse, underlyingError: CrossOriginRedirect())
                }
            }
            var fields = HTTPFields()
            for (name, value) in received.headers {
                guard let name = HTTPField.Name(name) else { continue }
                fields.append(HTTPField(name: name, value: value))
            }
            let response = HTTPResponse(
                status: HTTPResponse.Status(
                    code: Int(received.status.code),
                    reasonPhrase: received.status.reasonPhrase
                ),
                headerFields: fields
            )
            return try await responseHandler(
                response,
                HTTPResponseBody(received.body.map { Data($0.readableBytesView) })
            )
        }
    }

    /// The client followed a redirect to another origin.
    private struct CrossOriginRedirect: Error, CustomStringConvertible {
        var description: String {
            "The client followed a redirect to another origin, so its response wasn't read."
        }
    }
#endif
