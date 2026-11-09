/// An error that an embedding model throws.
///
/// Engines also throw their own errors,
/// such as `URLError` for a network failure,
/// or an error from the framework that loads a model.
public enum EmbeddingError: Error, Hashable, Sendable {
  /// The text at the index contains only whitespace.
  ///
  /// Models return vectors with no meaning for empty text,
  /// so every engine rejects it the same way.
  case emptyText(index: Int)

  /// The model isn't available, for the reason in the string.
  case unavailable(String)

  /// The model returned no vector for the text at the index.
  case noVector(index: Int)

  /// The model returned a different number of vectors than texts.
  case wrongVectorCount(expected: Int, actual: Int)

  /// The model returned a vector with a different number of elements than its ``EmbeddingModel/dimension``.
  case dimensionMismatch(expected: Int, actual: Int)

  /// The vector for the text at the index has zero length,
  /// or contains a value that isn't finite,
  /// so it can't be normalized.
  case invalidVector(index: Int)

  /// The string isn't a full, 40-character commit hash.
  case invalidCommit(String)

  /// A server returned an HTTP status that isn't a success,
  /// with the server's message.
  ///
  /// If the response might contain the API key,
  /// the message says that the server's text was withheld.
  case httpStatus(Int, message: String)

  /// A server returned a response that the model can't read.
  ///
  /// If the response might contain the API key,
  /// the description says that the server's text was withheld.
  case invalidResponse(String)
}
