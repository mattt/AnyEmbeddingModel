#if Llama
    import Foundation
    import Testing

    @testable import AnyEmbeddingModel

    extension Trait where Self == ConditionTrait {
        /// Runs a test only if `LLAMA_EMBEDDING_MODEL_PATH` is set.
        ///
        /// The tests expect `bge-small-en-v1.5-q8_0.gguf`
        /// from the Hugging Face Hub repository `ggml-org/bge-small-en-v1.5-Q8_0-GGUF`,
        /// at commit `f2068edd9b54f2a369549ccc71f70ed273a2a801`.
        static var needsLlamaEmbeddingModel: Self {
            .enabled(
                if: ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] != nil,
                "Set LLAMA_EMBEDDING_MODEL_PATH to the path of bge-small-en-v1.5-q8_0.gguf."
            )
        }

        /// Runs a test only if `LLAMA_MODEL_PATH` is set,
        /// the variable that AnyLanguageModel's Llama tests read.
        ///
        /// The test expects a GGUF model that generates text.
        static var needsLlamaLanguageModel: Self {
            .enabled(
                if: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] != nil,
                "Set LLAMA_MODEL_PATH to the path of a GGUF model that generates text."
            )
        }
    }

    @Suite("LlamaEmbeddingModel")
    struct LlamaEmbeddingModelTests {
        private func model(
            path: String = "/path/to/model.gguf",
            dimension: Int = 384,
            prefixes: EmbeddingPrefixes = .none,
            maximumTokenCount: Int = 512
        ) -> LlamaEmbeddingModel {
            LlamaEmbeddingModel(
                modelPath: path,
                name: "bge-small-en-v1.5-q8_0",
                version: "1",
                dimension: dimension,
                prefixes: prefixes,
                maximumTokenCount: maximumTokenCount
            )
        }

        @Test("The identifier has the name, the version, and the dimension")
        func identifier() {
            let base = model()
            #expect(base.identifier.hasPrefix("llama/bge-small-en-v1.5-q8_0/1/384/"))
            #expect(model(prefixes: .bge).identifier != base.identifier)
            #expect(model(maximumTokenCount: 256).identifier != base.identifier)
            // The path and the GPU layers don't change the vectors.
            #expect(model(path: "/elsewhere/model.gguf").identifier == base.identifier)
            let cpu = LlamaEmbeddingModel(
                modelPath: "/path/to/model.gguf",
                name: "bge-small-en-v1.5-q8_0",
                version: "1",
                dimension: 384,
                gpuLayers: 0
            )
            #expect(cpu.identifier == base.identifier)
            #expect(cpu.loaderKey != base.loaderKey)
            #expect(base.gpuLayers == LlamaEmbeddingModel.defaultGPULayerCount)
        }

        @Test("A missing file fails to load with an invalid path")
        func missingFile() async throws {
            let missing = model(path: "/nonexistent/\(UUID().uuidString).gguf")
            #expect(missing.availability == .unavailable(.notLoaded))
            await #expect(throws: LlamaEmbeddingModelError.invalidModelPath) {
                try await missing.embed("tea", for: .document)
            }
            guard case .unavailable(.failedToLoad) = missing.availability else {
                Issue.record("Expected a failed load, got \(missing.availability).")
                return
            }
        }

        @Test("An empty input returns no vectors without loading the model")
        func emptyInput() async throws {
            let missing = model(path: "/nonexistent/\(UUID().uuidString).gguf")
            #expect(try await missing.embed([], for: .document).isEmpty)
            #expect(missing.availability == .unavailable(.notLoaded))
            await #expect(throws: EmbeddingError.emptyText(index: 1)) {
                try await missing.embed(["tea", " "], for: .document)
            }
        }

        @Test("Groups fill each call up to the token and text limits, in order")
        func groups() {
            let groups = LlamaEmbeddingEngine.groups
            #expect(groups([], 10, 4) == [])
            #expect(groups([3, 3, 3], 10, 4) == [0 ..< 3])
            #expect(groups([3, 3, 3, 3], 10, 4) == [0 ..< 3, 3 ..< 4])
            #expect(groups([1, 1, 1, 1, 1], 10, 2) == [0 ..< 2, 2 ..< 4, 4 ..< 5])
            #expect(groups([10, 1, 10], 10, 4) == [0 ..< 1, 1 ..< 2, 2 ..< 3])
        }
    }

    /// Tests that load `bge-small-en-v1.5-q8_0.gguf` from `LLAMA_EMBEDDING_MODEL_PATH`.
    @Suite("LlamaEmbeddingModel with a model", .serialized, .needsLlamaEmbeddingModel)
    struct LlamaEmbeddingModelFileTests {
        static var path: String {
            ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] ?? ""
        }

        let model = LlamaEmbeddingModel(
            modelPath: path,
            name: "bge-small-en-v1.5-q8_0",
            version: "f2068edd9b54f2a369549ccc71f70ed273a2a801",
            dimension: 384,
            prefixes: .bge
        )

        private static let texts = [
            "My sister lives in Lisbon.",
            "The printer is out of toner.",
            "Green tea has less caffeine than coffee, but more than most herbal teas.",
            "Tea.",
            "The museum opens at nine on weekdays and at ten on weekends and public holidays.",
        ]

        @Test("Loading makes the model available and checks the dimension")
        func prewarm() async throws {
            try await model.prewarm()
            #expect(model.availability == .available)

            let wrong = LlamaEmbeddingModel(
                modelPath: Self.path,
                name: "bge-small-en-v1.5-q8_0",
                version: "f2068edd9b54f2a369549ccc71f70ed273a2a801",
                dimension: 768
            )
            await #expect(throws: EmbeddingError.dimensionMismatch(expected: 768, actual: 384)) {
                try await wrong.prewarm()
            }
            await #expect(throws: EmbeddingError.dimensionMismatch(expected: 768, actual: 384)) {
                try await wrong.embed("tea", for: .document)
            }
        }

        @Test("Vectors have the dimension and unit length")
        func unitVectors() async throws {
            let vectors = try await model.embed(Self.texts, for: .document)
            #expect(vectors.count == Self.texts.count)
            for vector in vectors {
                #expect(vector.count == 384)
                let length = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
                #expect(abs(length - 1) < 1e-5)
            }
        }

        @Test("The same text gives the same vector")
        func determinism() async throws {
            let first = try await model.embed(Self.texts[2], for: .query)
            let second = try await model.embed(Self.texts[2], for: .query)
            #expect(first == second)
        }

        @Test("A batch gives the same vectors as single texts")
        func batchMatchesSingle() async throws {
            // Two texts per call, so the batch takes three calls.
            let small = LlamaEmbeddingModel(
                modelPath: Self.path,
                name: "bge-small-en-v1.5-q8_0",
                version: "f2068edd9b54f2a369549ccc71f70ed273a2a801",
                dimension: 384,
                prefixes: .bge,
                maximumBatchSize: 2
            )
            for candidate in [model, small] {
                let batch = try await candidate.embed(Self.texts, for: .document)
                for (index, text) in Self.texts.enumerated() {
                    let single = try await candidate.embed(text, for: .document)
                    #expect(cosine(batch[index], single) > 0.9999)
                    let error = zip(batch[index], single).map { abs($0 - $1) }.max() ?? 0
                    #expect(error < 1e-3)
                }
            }
        }

        @Test("A related pair scores above an unrelated pair")
        func semantics() async throws {
            let query = try await model.embed("Where does my sister live?", for: .query)
            let documents = try await model.embed(Self.texts, for: .document)
            let related = cosine(query, documents[0])
            let unrelated = cosine(query, documents[1])
            #expect(related > unrelated + 0.1)
        }

        @Test("A long text is truncated to the token limit, keeping the last token")
        func truncation() async throws {
            let limited = LlamaEmbeddingModel(
                modelPath: Self.path,
                name: "bge-small-en-v1.5-q8_0",
                version: "f2068edd9b54f2a369549ccc71f70ed273a2a801",
                dimension: 384,
                maximumTokenCount: 16
            )
            let shared = Array(repeating: "tea", count: 40).joined(separator: " ")
            let texts = [shared + " and coffee", shared + " and juice"]
            let tokens = try await limited.tokenIDs(for: texts, purpose: .document)
            #expect(tokens.map(\.count) == [16, 16])
            #expect(tokens[0] == tokens[1])
            // BERT's [CLS] and [SEP].
            #expect(tokens[0].first == 101)
            #expect(tokens[0].last == 102)

            let vectors = try await limited.embed(texts, for: .document)
            #expect(vectors[0] == vectors[1])

            // The full model reads the tails, so its vectors differ.
            let full = try await model.embed(texts, for: .document)
            #expect(full[0] != full[1])
        }

        @Test("Tokens match sentence-transformers, and vectors agree with its reference")
        func reference() async throws {
            let fixture = try BGEReference.load()
            // The fixture has 11 cases, from short queries to texts above the token limit.
            #expect(fixture.cases.count == 11)
            for item in fixture.cases {
                let purpose: EmbeddingPurpose = item.purpose == "query" ? .query : .document
                let tokens = try await model.tokenIDs(for: [item.text], purpose: purpose)
                #expect(tokens[0].map(Int.init) == item.tokens, "\(item.name)")
                let vector = try await model.embed(item.text, for: purpose)
                // The GGUF weights are quantized to 8 bits.
                // The lowest observed cosine similarity is 0.99977,
                // and the largest element error is 0.0031.
                #expect(cosine(vector, item.vector) > 0.9995, "\(item.name)")
                let error = zip(vector, item.vector).map { abs($0 - $1) }.max() ?? 0
                #expect(error < 0.005, "\(item.name)")
            }
        }

        @Test("Concurrent embeddings give the same vectors")
        func concurrency() async throws {
            let expected = try await model.embed(Self.texts, for: .document)
            #expect(expected.count == Self.texts.count)
            let model = model
            let results = try await withThrowingTaskGroup(of: [[Float]].self) { group in
                for _ in 0 ..< 4 {
                    group.addTask { try await model.embed(Self.texts, for: .document) }
                }
                return try await group.reduce(into: []) { $0.append($1) }
            }
            #expect(results.count == 4)
            for vectors in results {
                #expect(vectors.count == expected.count)
                for (actual, reference) in zip(vectors, expected) {
                    #expect(cosine(actual, reference) > 0.9999)
                }
            }
        }

        @Test("Removing the model from the cache unloads it, and the next embedding loads it")
        func removeFromCache() async throws {
            try await model.prewarm()
            await model.removeFromCache()
            #expect(model.availability == .unavailable(.notLoaded))
            _ = try await model.embed("tea", for: .document)
            #expect(model.availability == .available)
        }
    }

    /// Tests that exit a child process with a model loaded on the GPU.
    ///
    /// Without the package's exit handler,
    /// ggml's Metal backend stops such a process with an assertion,
    /// and the child exits with `SIGABRT` instead of success.
    @Suite("LlamaEmbeddingModel at exit", .needsLlamaEmbeddingModel)
    struct LlamaEmbeddingModelExitTests {
        @Test("A process exits normally with a model loaded and idle")
        func idleExit() async {
            await #expect(processExitsWith: .success) {
                let model = LlamaEmbeddingModel(
                    modelPath: ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] ?? "",
                    name: "bge-small-en-v1.5-q8_0",
                    version: "1",
                    dimension: 384
                )
                _ = try await model.embed("tea", for: .document)
                exit(0)
            }
        }

        @Test("A process exits normally during an embedding")
        func exitDuringEmbedding() async {
            await #expect(processExitsWith: .success) {
                let model = LlamaEmbeddingModel(
                    modelPath: ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] ?? "",
                    name: "bge-small-en-v1.5-q8_0",
                    version: "1",
                    dimension: 384
                )
                try await model.prewarm()
                let texts = (0 ..< 20_000).map { "Note \($0): green tea has less caffeine than coffee." }
                Task.detached { _ = try? await model.embed(texts, for: .document) }
                // Wait until the embedding runs, then exit in the middle of it.
                let deadline = Date().addingTimeInterval(10)
                while LlamaLifecycle.shared.activeOperationCount == 0 {
                    // A distinct status shows that the embedding never started.
                    guard Date() < deadline else { exit(3) }
                    usleep(1_000)
                }
                usleep(50_000)
                guard LlamaLifecycle.shared.activeOperationCount > 0 else { exit(4) }
                exit(0)
            }
        }

        @Test("After the exit handler runs, an embedding throws and nothing loads")
        func workAfterShutdown() async {
            await #expect(processExitsWith: .success) {
                let model = LlamaEmbeddingModel(
                    modelPath: ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] ?? "",
                    name: "bge-small-en-v1.5-q8_0",
                    version: "1",
                    dimension: 384
                )
                _ = try await model.embed("tea", for: .document)
                LlamaLifecycle.shared.shutDown()
                do {
                    _ = try await model.embed("tea", for: .document)
                    exit(5)
                } catch let error as EmbeddingError where error.kind == .unavailable {
                    exit(0)
                }
            }
        }
    }

    @Suite("LlamaEmbeddingModel with a language model", .needsLlamaLanguageModel)
    struct LlamaEmbeddingModelLanguageModelTests {
        @Test("A model that generates text fails to load")
        func rejectsLanguageModel() async throws {
            let model = LlamaEmbeddingModel(
                modelPath: ProcessInfo.processInfo.environment["LLAMA_MODEL_PATH"] ?? "",
                name: "language-model",
                version: "1",
                dimension: 64
            )
            await #expect(throws: LlamaEmbeddingModelError.notEmbeddingModel) {
                try await model.prewarm()
            }
        }
    }

    /// The sentence-transformers vectors for `BAAI/bge-small-en-v1.5`
    /// that `mise run generate:fixtures` wrote for the MLX tests.
    private struct BGEReference: Decodable {
        struct Case: Decodable {
            let name: String
            let purpose: String
            let text: String
            let tokens: [Int]
            let vector: [Float]
        }

        let cases: [Case]

        static func load() throws -> BGEReference {
            let url = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .appendingPathComponent("Fixtures/reference-bge-small.json")
            return try JSONDecoder().decode(BGEReference.self, from: Data(contentsOf: url))
        }
    }
#endif
