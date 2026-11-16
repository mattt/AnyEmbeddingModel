#if MLX
    import Foundation
    import Testing

    @testable import AnyEmbeddingModel

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

    extension Trait where Self == ConditionTrait {
        /// Runs a test only if `ANY_EMBEDDING_MODEL_TEST_METAL` is set.
        ///
        /// Loading a model or setting MLX's cache limit starts MLX's Metal device,
        /// which stops the process if it can't find MLX's Metal library.
        /// The `test:mlx` task puts the library where the test process finds it
        /// and sets the variable.
        static var needsMetalLibrary: Self {
            .enabled(
                if: ProcessInfo.processInfo.environment["ANY_EMBEDDING_MODEL_TEST_METAL"] != nil,
                "Run `mise run test:mlx`, which puts MLX's Metal library where the tests find it."
            )
        }
    }

    /// Runs `body`, then removes each model from the cache, even if `body` throws.
    func removingFromCache<Result>(
        _ models: MLXEmbeddingModel...,
        body: () async throws -> Result
    ) async throws -> Result {
        do {
            let result = try await body()
            for model in models { await model.removeFromCache() }
            return result
        } catch {
            for model in models { await model.removeFromCache() }
            throw error
        }
    }

    /// Returns the cosine similarity of two vectors.
    func cosine(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
            / (a.reduce(0) { $0 + $1 * $1 }.squareRoot() * b.reduce(0) { $0 + $1 * $1 }.squareRoot())
    }
#endif
