import Foundation
import HTTPTypes

/// Why a request to a model's server failed:
/// it couldn't be made, its answer wasn't an HTTP response,
/// the server answered with an error status,
/// or a successful answer couldn't be read.
///
/// A caller can read an error response's status, headers, and body,
/// for example to wait as its `Retry-After` header says.
///
/// A request that fails on the way, such as one the network drops,
/// throws the transport's own error instead:
/// `URLError` from a URL session, or `HTTPClientError` from AsyncHTTPClient.
/// So does a cancelled request.
///
/// This type has the same shape as AnyLanguageModel's `HTTPTransportError`.
public struct HTTPTransportError: Error, Sendable, CustomStringConvertible {
    /// What went wrong. More kinds can be added, so a `switch` over them needs a `default`.
    public struct Kind: Hashable, Sendable, CustomStringConvertible {
        public let description: String

        private init(_ description: String) {
            self.description = description
        }

        /// The request couldn't be sent as given.
        public static let invalidRequest = Kind("Invalid request")
        /// The answer wasn't an HTTP response.
        public static let invalidResponse = Kind("Invalid response")
        /// The server answered with an error status: the error has its response and body.
        public static let httpError = Kind("HTTP error")
        /// A successful answer couldn't be read as what was expected.
        public static let decodingError = Kind("Decoding error")
    }

    /// What went wrong.
    public let kind: Kind

    /// The response, headers included, when the server answered.
    public let response: HTTPResponse?

    /// The response's body, when it was read.
    public let body: Data?

    /// The error that caused this one, such as a decoding error's.
    public let underlyingError: (any Error)?

    public init(
        kind: Kind,
        response: HTTPResponse? = nil,
        body: Data? = nil,
        underlyingError: (any Error)? = nil
    ) {
        self.kind = kind
        self.response = response
        self.body = body
        self.underlyingError = underlyingError
    }

    public var description: String {
        var description = kind.description
        if let response { description += " (Status \(response.status.code))" }
        if let body, let text = String(data: body, encoding: .utf8), !text.isEmpty { description += ": \(text)" }
        if let underlyingError { description += ": \(underlyingError)" }
        return description
    }
}

/// What every transport's requests to a model's server are made of,
/// written once on top of ``HTTPTransport/send(request:body:timeout:responseHandler:)``.
extension HTTPTransport {
    /// Sends `request` with `body`, and returns the whole body of a successful response.
    ///
    /// - Throws: ``HTTPTransportError`` with ``HTTPTransportError/Kind/httpError``,
    ///   the response, and its body, if the status isn't a success,
    ///   or the transport's error if the request fails or is cancelled.
    func fetch(_ request: HTTPRequest, body: Data? = nil) async throws -> Data {
        let request = Self.request(request, accepting: "application/json", body: body)
        let data = try await send(request: request, body: body, timeout: nil) { response, body in
            try await Self.validate(response, body)
            return try await body.collect()
        }
        // A body cut short by cancelling isn't read as a whole one, whichever transport sent it.
        try Task.checkCancellation()
        return data
    }

    /// `request`, accepting `accept` and saying it sends JSON when it has `body`,
    /// unless it says otherwise.
    private static func request(_ request: HTTPRequest, accepting accept: String, body: Data?) -> HTTPRequest {
        var request = request
        if request.headerFields[.accept] == nil { request.headerFields[.accept] = accept }
        if body != nil, request.headerFields[.contentType] == nil {
            request.headerFields[.contentType] = "application/json"
        }
        return request
    }

    /// A response that isn't a success fails with what its body says.
    private static func validate(_ response: HTTPResponse, _ body: HTTPResponseBody) async throws {
        guard response.status.kind == .successful else {
            throw HTTPTransportError(kind: .httpError, response: response, body: try await body.collect())
        }
    }
}
