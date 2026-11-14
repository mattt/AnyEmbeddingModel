/// An error that an embedding model throws.
///
/// Check ``kind`` to decide what to do,
/// and read the properties that the kind describes for details,
/// such as ``index`` for the text that failed.
///
/// A request to a server that fails throws ``HTTPTransportError`` instead,
/// and a request that fails on the way throws the transport's own error,
/// such as `URLError`.
/// Engines also throw their own errors,
/// such as an error from the framework that loads a model,
/// and a cancelled call throws `CancellationError` or the transport's error.
public struct EmbeddingError: Error, Hashable, Sendable, CustomStringConvertible {
    /// What went wrong. More kinds can be added, so a `switch` over them needs a `default`.
    public struct Kind: Hashable, Sendable, CustomStringConvertible {
        public let description: String

        private init(_ description: String) {
            self.description = description
        }

        /// The text at ``EmbeddingError/index`` contains only whitespace.
        ///
        /// Models return vectors with no meaning for empty text,
        /// so every engine rejects it the same way.
        public static let emptyText = Kind("Empty text")

        /// The model isn't available, for the reason in ``EmbeddingError/message``.
        public static let unavailable = Kind("Unavailable")

        /// The model returned no vector for the text at ``EmbeddingError/index``.
        public static let noVector = Kind("No vector")

        /// The model returned ``EmbeddingError/actual`` vectors
        /// for ``EmbeddingError/expected`` texts.
        public static let wrongVectorCount = Kind("Wrong vector count")

        /// The model returned a vector with ``EmbeddingError/actual`` elements,
        /// and its ``EmbeddingModel/dimension`` is ``EmbeddingError/expected``.
        public static let dimensionMismatch = Kind("Dimension mismatch")

        /// The vector for the text at ``EmbeddingError/index`` has zero length,
        /// or contains a value that isn't finite,
        /// so it can't be normalized.
        public static let invalidVector = Kind("Invalid vector")

        /// The string in ``EmbeddingError/message`` isn't a full, 40-character commit hash.
        public static let invalidCommit = Kind("Invalid commit")

        /// The string in ``EmbeddingError/message`` isn't the name of a Hub repository
        /// of the form `namespace/name`.
        public static let invalidRepository = Kind("Invalid repository")

        /// A model or a server returned something that the model can't use,
        /// as ``EmbeddingError/message`` describes.
        public static let invalidResponse = Kind("Invalid response")
    }

    /// What went wrong.
    public let kind: Kind

    /// The index of the text in the caller's input that the error is about, if any.
    public let index: Int?

    /// The number that the model expected, for a count or a dimension.
    public let expected: Int?

    /// The number that the model received, for a count or a dimension.
    public let actual: Int?

    /// A description of the problem, or the value that was rejected, if any.
    public let message: String?

    private init(
        _ kind: Kind,
        index: Int? = nil,
        expected: Int? = nil,
        actual: Int? = nil,
        message: String? = nil
    ) {
        self.kind = kind
        self.index = index
        self.expected = expected
        self.actual = actual
        self.message = message
    }

    /// An error of kind ``Kind/emptyText`` for the text at `index`.
    public static func emptyText(index: Int) -> EmbeddingError {
        EmbeddingError(.emptyText, index: index)
    }

    /// An error of kind ``Kind/unavailable``, with the reason.
    public static func unavailable(_ reason: String) -> EmbeddingError {
        EmbeddingError(.unavailable, message: reason)
    }

    /// An error of kind ``Kind/noVector`` for the text at `index`.
    public static func noVector(index: Int) -> EmbeddingError {
        EmbeddingError(.noVector, index: index)
    }

    /// An error of kind ``Kind/wrongVectorCount``.
    public static func wrongVectorCount(expected: Int, actual: Int) -> EmbeddingError {
        EmbeddingError(.wrongVectorCount, expected: expected, actual: actual)
    }

    /// An error of kind ``Kind/dimensionMismatch``.
    public static func dimensionMismatch(expected: Int, actual: Int) -> EmbeddingError {
        EmbeddingError(.dimensionMismatch, expected: expected, actual: actual)
    }

    /// An error of kind ``Kind/invalidVector`` for the text at `index`.
    public static func invalidVector(index: Int) -> EmbeddingError {
        EmbeddingError(.invalidVector, index: index)
    }

    /// An error of kind ``Kind/invalidCommit`` for the rejected string.
    public static func invalidCommit(_ commit: String) -> EmbeddingError {
        EmbeddingError(.invalidCommit, message: commit)
    }

    /// An error of kind ``Kind/invalidRepository`` for the rejected string.
    public static func invalidRepository(_ modelId: String) -> EmbeddingError {
        EmbeddingError(.invalidRepository, message: modelId)
    }

    /// An error of kind ``Kind/invalidResponse``, with a description of the problem.
    public static func invalidResponse(_ description: String) -> EmbeddingError {
        EmbeddingError(.invalidResponse, message: description)
    }

    public var description: String {
        var description = kind.description
        if let index { description += " at index \(index)" }
        if let expected, let actual { description += ": expected \(expected), got \(actual)" }
        if let message { description += ": \(message)" }
        return description
    }
}
