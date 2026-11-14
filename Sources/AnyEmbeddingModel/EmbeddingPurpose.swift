/// The use of a text that an ``EmbeddingModel`` embeds.
///
/// Many retrieval models embed queries and documents differently.
/// Most add a prefix to the text,
/// such as `"search_query: "` and `"search_document: "`,
/// and some add an instruction only to queries.
/// A model uses the purpose to choose the prefix,
/// so that callers don't need to know the convention of each model.
public enum EmbeddingPurpose: Sendable, Hashable, CaseIterable {
    /// Text to search with, such as a question.
    case query

    /// Text to search for, such as a saved note or a passage.
    case document
}
