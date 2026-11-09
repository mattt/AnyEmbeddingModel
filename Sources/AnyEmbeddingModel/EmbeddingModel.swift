/// A type that turns text into vectors for semantic search.
///
/// An embedding model returns one vector for each text that you give it.
/// Texts with similar meanings get vectors with a high cosine similarity.
/// Because every vector is unit length,
/// the dot product of two vectors is their cosine similarity.
///
/// ```swift
/// let model = NaturalLanguageEmbeddingModel(language: .english)!
/// let documents = try await model.embed(["The cat sat on the mat."], for: .document)
/// let query = try await model.embed("Where did the cat sit?", for: .query)
/// ```
///
/// ## Conforming to the protocol
///
/// A conforming type must keep these promises,
/// which callers can depend on without knowing the engine:
///
/// - The batch `embed(_:for:)` returns one vector for each input,
///   in the same order as the inputs.
/// - Each vector has ``dimension`` elements,
///   all finite, with a Euclidean length of 1.
/// - The model applies its own prefixes or instructions for each ``EmbeddingPurpose``.
///   Callers pass the plain text.
/// - ``identifier`` changes whenever a vector for the same text and purpose could change.
public protocol EmbeddingModel: Sendable {
  /// The type that describes why the model can't embed text now.
  ///
  /// A model that is always available uses `Never`.
  associatedtype UnavailableReason: Sendable = Never

  /// A string that identifies the vectors that this model produces.
  ///
  /// Two models with the same identifier
  /// return the same vectors for the same text and purpose,
  /// within floating-point error.
  /// The identifier includes the engine, the model, its revision,
  /// the dimension, and every setting that changes vectors,
  /// such as prefixes and truncation.
  /// Store it with saved vectors,
  /// and embed the texts again when it changes.
  ///
  /// An identifier is ASCII text with five fields, separated by slashes:
  /// the engine, the model, the revision, the dimension, and a fingerprint of the other settings.
  /// For example:
  /// `mlx/BAAI%2Fbge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/384/17ad23889b01fd7d`.
  /// Each field is the percent-encoding of its UTF-8 bytes,
  /// and a revision of `-` means that the model has none.
  ///
  /// Identity is by bytes, with no Unicode normalization:
  /// composed and decomposed forms of the same text give different identifiers,
  /// because they can give different vectors.
  var identifier: String { get }

  /// The number of elements in each vector.
  var dimension: Int { get }

  /// Whether the model can embed text now.
  var availability: Availability<UnavailableReason> { get }

  /// Loads the model and its resources before the first embedding.
  ///
  /// The batch `embed(_:for:)` loads the model if it isn't loaded,
  /// so you don't have to call this method.
  /// Call it to move the cost of a download or a load
  /// to a time that you choose.
  func prewarm() async throws

  /// Returns a vector for each text.
  ///
  /// - Parameters:
  ///   - texts: The texts to embed.
  ///     An empty array returns an empty array.
  ///   - purpose: Whether the texts are queries or documents.
  /// - Returns: One unit vector of ``dimension`` elements for each text,
  ///   in the order of `texts`.
  /// - Throws: ``EmbeddingError/emptyText(index:)``
  ///   if a text contains only whitespace,
  ///   or another ``EmbeddingError`` or an engine error if the model fails.
  ///   If the model can't embed one text, it throws and returns no vectors.
  func embed(_ texts: [String], for purpose: EmbeddingPurpose) async throws -> [[Float]]
}

extension EmbeddingModel {
  /// Returns the vector for one text.
  ///
  /// - Parameters:
  ///   - text: The text to embed.
  ///   - purpose: Whether the text is a query or a document.
  public func embed(_ text: String, for purpose: EmbeddingPurpose) async throws -> [Float] {
    let vectors = try await embed([text], for: purpose)
    guard let vector = vectors.first, vectors.count == 1 else {
      throw EmbeddingError.wrongVectorCount(expected: 1, actual: vectors.count)
    }
    return vector
  }

  /// A Boolean value that indicates whether the model can embed text now.
  public var isAvailable: Bool {
    if case .available = availability { true } else { false }
  }

  /// Does nothing, for a model that has nothing to load.
  public func prewarm() async throws {}
}

extension EmbeddingModel where UnavailableReason == Never {
  /// Returns ``Availability/available``, for a model that is always available.
  public var availability: Availability<UnavailableReason> { .available }
}
