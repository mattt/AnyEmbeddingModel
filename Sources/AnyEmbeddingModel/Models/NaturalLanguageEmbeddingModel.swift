#if canImport(NaturalLanguage)
  import Foundation
  import NaturalLanguage

  /// An embedding model that uses the sentence embedding of the NaturalLanguage framework.
  ///
  /// `NLEmbedding.sentenceEmbedding(for:)` is part of the operating system,
  /// so this model needs no download and no dependency.
  /// It has one vector form for all texts,
  /// so it embeds queries and documents the same way
  /// unless you give it prefixes.
  ///
  /// The framework doesn't document a limit on the length of a text
  /// or say how it treats a long text.
  /// Use this model for short texts, such as sentences and notes.
  ///
  /// ```swift
  /// guard let model = NaturalLanguageEmbeddingModel(language: .english) else {
  ///   return  // The system has no sentence embedding for the language.
  /// }
  /// let vector = try await model.embed("I like tea.", for: .document)
  /// ```
  public struct NaturalLanguageEmbeddingModel: EmbeddingModel {
    /// The language of the texts.
    public let language: NLLanguage

    /// The revision of the system's sentence embedding.
    public let revision: Int

    /// The number of elements in each vector.
    public let dimension: Int

    /// The text that the model puts before each input.
    public let prefixes: EmbeddingPrefixes

    /// A string that identifies the vectors that this model produces.
    ///
    /// The identifier has the language, ``revision``, ``dimension``,
    /// and a fingerprint of ``prefixes``.
    /// If you don't pin a revision and an update to the system changes the current one,
    /// the identifier changes too.
    public let identifier: String

    private let embedding: Locked<NLEmbedding>

    /// Creates a model with the current sentence embedding for a language,
    /// or returns `nil` if the system has none.
    ///
    /// - Parameters:
    ///   - language: The language of the queries and documents.
    ///   - revision: A revision from `NLEmbedding.supportedSentenceEmbeddingRevisions(for:)`
    ///     to keep the same vectors across system updates,
    ///     or `nil` for the current revision.
    ///   - prefixes: The text before each query and document.
    ///     The default adds nothing.
    public init?(
      language: NLLanguage,
      revision: Int? = nil,
      prefixes: EmbeddingPrefixes = .none
    ) {
      let embedding =
        if let revision {
          NLEmbedding.sentenceEmbedding(for: language, revision: revision)
        } else {
          NLEmbedding.sentenceEmbedding(for: language)
        }
      guard let embedding else { return nil }
      self.language = language
      self.revision = embedding.revision
      self.dimension = embedding.dimension
      self.prefixes = prefixes
      self.embedding = Locked(embedding)
      self.identifier = Identifier.make(
        engine: "naturallanguage",
        model: "sentence-\(language.rawValue)",
        revision: String(embedding.revision),
        dimension: embedding.dimension,
        settings: [prefixes.query, prefixes.document]
      )
    }

    /// Returns a vector for each text.
    ///
    /// The framework embeds one text at a time,
    /// so a batch takes as long as its texts one after another.
    public func embed(_ texts: [String], for purpose: EmbeddingPurpose) async throws -> [[Float]] {
      try Vectors.validate(texts)
      var vectors: [[Float]] = []
      vectors.reserveCapacity(texts.count)
      for (index, text) in texts.enumerated() {
        try Task.checkCancellation()
        // The framework doesn't say that one embedding is safe to use from two threads.
        let values = embedding.withLock {
          $0.vector(for: prefixes.apply(to: text, for: purpose))
        }
        guard let values else { throw EmbeddingError.noVector(index: index) }
        let vector = try Vectors.normalized(
          values.map(Float.init),
          dimension: dimension,
          index: index
        )
        vectors.append(vector)
      }
      return vectors
    }
  }
#endif
