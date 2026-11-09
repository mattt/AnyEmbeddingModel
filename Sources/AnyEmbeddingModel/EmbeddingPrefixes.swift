/// The text that a model puts before each input, for each purpose.
///
/// Retrieval models are trained with a fixed prefix for queries,
/// for documents, or for both.
/// A model that doesn't get its prefixes gives worse results,
/// and a model that gets the wrong prefixes can give much worse results.
/// Use the preset for a model when one exists.
///
/// The model adds the prefix exactly as written,
/// with no space or separator,
/// so include any space or newline in the prefix.
public struct EmbeddingPrefixes: Sendable, Hashable, Codable {
  /// The text before each query.
  public var query: String

  /// The text before each document.
  public var document: String

  /// Creates prefixes.
  public init(query: String = "", document: String = "") {
    self.query = query
    self.document = document
  }

  /// No prefixes, for models that embed queries and documents the same way.
  public static let none = EmbeddingPrefixes()

  /// The prefixes of the BGE English v1.5 models.
  ///
  /// These models add an instruction to short queries and nothing to documents.
  public static let bge = EmbeddingPrefixes(
    query: "Represent this sentence for searching relevant passages: "
  )

  /// The prefixes of the E5 models.
  public static let e5 = EmbeddingPrefixes(query: "query: ", document: "passage: ")

  /// The prefixes of the Nomic Embed Text models.
  public static let nomic = EmbeddingPrefixes(
    query: "search_query: ",
    document: "search_document: "
  )

  /// The prefixes of the Qwen3 Embedding models, with a task description.
  ///
  /// These models put an instruction before each query and nothing before documents.
  ///
  /// - Parameter task: A one-sentence description of the retrieval task.
  public static func qwen3(
    task: String = "Given a web search query, retrieve relevant passages that answer the query"
  ) -> EmbeddingPrefixes {
    EmbeddingPrefixes(query: "Instruct: \(task)\nQuery:")
  }

  /// Returns the prefix for a purpose.
  public subscript(purpose: EmbeddingPurpose) -> String {
    switch purpose {
    case .query: query
    case .document: document
    }
  }

  /// Returns `text` with the prefix for `purpose` before it.
  public func apply(to text: String, for purpose: EmbeddingPurpose) -> String {
    self[purpose] + text
  }
}
