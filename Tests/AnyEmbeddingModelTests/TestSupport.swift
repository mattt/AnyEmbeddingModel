#if MLX
  import Foundation
  import Testing

  extension Trait where Self == ConditionTrait {
    /// Runs a test only if `ANY_EMBEDDING_MODEL_TEST_HUB` is set.
    ///
    /// These tests load real weights.
    /// They download the models from the Hugging Face Hub on the first run,
    /// and they can run offline after that, from the cache.
    static var needsRealModels: Self {
      .enabled(
        if: ProcessInfo.processInfo.environment["ANY_EMBEDDING_MODEL_TEST_HUB"] != nil,
        "Set ANY_EMBEDDING_MODEL_TEST_HUB=1 to load real models."
      )
    }
  }

  /// Returns the cosine similarity of two vectors.
  func cosine(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
      / (a.reduce(0) { $0 + $1 * $1 }.squareRoot() * b.reduce(0) { $0 + $1 * $1 }.squareRoot())
  }
#endif
