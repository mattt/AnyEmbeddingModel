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

    // The exit tests use test hooks, which only debug builds compile.
    #if DEBUG
        /// Tests that exit a child process with a model loaded on the GPU.
        ///
        /// Without the package's exit handler,
        /// ggml's Metal backend stops such a process with an assertion,
        /// and the child exits with `SIGABRT` instead of success.
        /// Each test asserts the exact status of its child,
        /// so that a setup failure, which exits with its own status, can't pass.
        @Suite("LlamaEmbeddingModel at exit", .needsLlamaEmbeddingModel)
        struct LlamaEmbeddingModelExitTests {
            /// Returns a model that runs one text per call, after its first embedding.
            static func loadedModel() async throws -> LlamaEmbeddingModel {
                let model = LlamaEmbeddingModel(
                    modelPath: ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] ?? "",
                    name: "bge-small-en-v1.5-q8_0",
                    version: "1",
                    dimension: 384,
                    maximumBatchSize: 1
                )
                _ = try await model.embed("tea", for: .document)
                return model
            }

            static let texts = ["Green tea.", "Black tea.", "Coffee.", "Water."]

            /// Holds an embedding until the exit handler has started,
            /// so that the handler must wait for it.
            static func holdUntilShutdownStarts() {
                let limit = ExitTestTimer.now() + 10_000_000_000
                while !LlamaLifecycle.shared.isStopping {
                    // A distinct status shows that the exit handler never started.
                    if ExitTestTimer.now() > limit { _exit(9) }
                    usleep(1_000)
                }
            }

            /// Starts an embedding in another task,
            /// and returns when its first call to llama.cpp is about to start.
            ///
            /// - Parameter whileActive: What the embedding does before that call,
            ///   while it is known to be running.
            static func startEmbedding(
                _ model: LlamaEmbeddingModel,
                whileActive: @escaping @Sendable () -> Void
            ) {
                let started = DispatchSemaphore(value: 0)
                let once = Locked(false)
                LlamaTestHooks.beforeDecode.withLock {
                    $0 = {
                        let isFirst = once.withLock { done in
                            defer { done = true }
                            return !done
                        }
                        guard isFirst else { return }
                        started.signal()
                        whileActive()
                    }
                }
                Task.detached { _ = try? await model.embed(texts, for: .document) }
                // A distinct status shows that the embedding never started.
                if started.wait(timeout: .now() + 10) == .timedOut { _exit(3) }
            }

            @Test("A process exits normally with a model loaded and idle")
            func idleExit() async throws {
                let watchdog = ExitTestTimer.watch("idleExit")
                await #expect(processExitsWith: .success) {
                    ExitTestTimer.childStarted("idleExit")
                    _ = try await LlamaEmbeddingModelExitTests.loadedModel()
                    ExitTestTimer.stepStarted("idleExit")
                    exit(0)
                }
                let elapsed = await ExitTestTimer.finish("idleExit", watchdog)
                #expect(try #require(elapsed) < .seconds(LlamaLifecycle.exitDeadline))
            }

            @Test("The exit handler waits for an embedding that is running, and the process exits normally")
            func exitDuringEmbedding() async throws {
                let watchdog = ExitTestTimer.watch("exitDuringEmbedding")
                await #expect(processExitsWith: .success) {
                    ExitTestTimer.childStarted("exitDuringEmbedding")
                    let model = try await LlamaEmbeddingModelExitTests.loadedModel()
                    LlamaEmbeddingModelExitTests.startEmbedding(model) {
                        LlamaEmbeddingModelExitTests.holdUntilShutdownStarts()
                    }
                    ExitTestTimer.stepStarted("exitDuringEmbedding")
                    exit(0)
                }
                let elapsed = await ExitTestTimer.finish("exitDuringEmbedding", watchdog)
                #expect(try #require(elapsed) < .seconds(LlamaLifecycle.exitDeadline))
            }

            @Test("An embedding that runs past the deadline doesn't keep the process alive")
            func exitPastDeadline() async throws {
                let watchdog = ExitTestTimer.watch("exitPastDeadline")
                // The handler frees nothing, so ggml's assertion ends the process,
                // as it would without the handler.
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitPastDeadline")
                    let model = try await LlamaEmbeddingModelExitTests.loadedModel()
                    LlamaEmbeddingModelExitTests.startEmbedding(model) {
                        while true { sleep(60) }
                    }
                    ExitTestTimer.stepStarted("exitPastDeadline")
                    exit(0)
                }
                let elapsed = try #require(await ExitTestTimer.finish("exitPastDeadline", watchdog))
                #expect(elapsed >= .seconds(LlamaLifecycle.exitDeadline))
                #expect(elapsed < .seconds(LlamaLifecycle.exitDeadline + 5))
            }

            @Test("An exit from inside an embedding doesn't wait for that embedding")
            func exitInsideOperation() async throws {
                let watchdog = ExitTestTimer.watch("exitInsideOperation")
                // The handler returns at once and frees nothing,
                // so ggml's assertion ends the process before the deadline.
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitInsideOperation")
                    let model = try await LlamaEmbeddingModelExitTests.loadedModel()
                    LlamaEmbeddingModelExitTests.startEmbedding(model) {
                        ExitTestTimer.stepStarted("exitInsideOperation")
                        exit(0)
                    }
                    // The embedding's thread exits the process.
                    while true { sleep(60) }
                }
                let elapsed = await ExitTestTimer.finish("exitInsideOperation", watchdog)
                #expect(try #require(elapsed) < .seconds(LlamaLifecycle.exitDeadline))
            }

            @Test("A log callback that exits while a model is freed doesn't hang the process")
            func exitFromLogDuringCleanup() async throws {
                // The handler sees that the cleanup's thread is exiting, and returns at once.
                // The model is only partly freed, so ggml's assertion ends the process,
                // as it would without the handler.
                let watchdog = ExitTestTimer.watch("exitFromLogDuringCleanup")
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("exitFromLogDuringCleanup")
                    let model = try await LlamaEmbeddingModelExitTests.loadedModel()
                    let once = Locked(false)
                    LlamaTestHooks.onLog.withLock {
                        $0 = {
                            let isFirst = once.withLock { done in
                                defer { done = true }
                                return !done
                            }
                            if isFirst { exit(0) }
                        }
                    }
                    // Unloading frees the model, and llama.cpp logs while it frees.
                    ExitTestTimer.stepStarted("exitFromLogDuringCleanup")
                    await model.removeFromCache()
                    // A distinct status shows that llama.cpp logged nothing.
                    exit(4)
                }
                let elapsed = await ExitTestTimer.finish("exitFromLogDuringCleanup", watchdog)
                #expect(try #require(elapsed) < .seconds(LlamaLifecycle.exitDeadline))
            }

            @Test("After the exit handler runs, a loaded model is unavailable, and embed and prewarm throw")
            func workAfterShutdown() async {
                let watchdog = ExitTestTimer.watch("workAfterShutdown")
                await #expect(processExitsWith: .success) {
                    ExitTestTimer.childStarted("workAfterShutdown")
                    let model = try await LlamaEmbeddingModelExitTests.loadedModel()
                    guard LlamaLifecycle.shared.shutDown() == .freed else { exit(6) }
                    guard model.availability == .unavailable(.exiting) else { exit(7) }
                    do {
                        _ = try await model.embed("tea", for: .document)
                        exit(5)
                    } catch let error as EmbeddingError where error.kind == .unavailable {}
                    do {
                        try await model.prewarm()
                        exit(8)
                    } catch let error as EmbeddingError where error.kind == .unavailable {}
                    exit(0)
                }
                _ = await ExitTestTimer.finish("workAfterShutdown", watchdog)
            }

            @Test("After the exit handler runs, a model that isn't loaded fails to load")
            func loadAfterShutdown() async {
                let watchdog = ExitTestTimer.watch("loadAfterShutdown")
                await #expect(processExitsWith: .success) {
                    ExitTestTimer.childStarted("loadAfterShutdown")
                    _ = try await LlamaEmbeddingModelExitTests.loadedModel()
                    LlamaLifecycle.shared.shutDown()
                    // Another version has its own loader, so it loads the file again.
                    let other = LlamaEmbeddingModel(
                        modelPath: ProcessInfo.processInfo.environment["LLAMA_EMBEDDING_MODEL_PATH"] ?? "",
                        name: "bge-small-en-v1.5-q8_0",
                        version: "2",
                        dimension: 384
                    )
                    do {
                        try await other.prewarm()
                        exit(5)
                    } catch let error as EmbeddingError where error.kind == .unavailable {}
                    guard other.availability == .unavailable(.exiting) else { exit(6) }
                    // The exit error comes before every other check of a request:
                    // a missing file, an empty input, and whitespace.
                    let missing = LlamaEmbeddingModel(
                        modelPath: "/nonexistent/model.gguf",
                        name: "missing",
                        version: "1",
                        dimension: 384
                    )
                    do {
                        try await missing.prewarm()
                        exit(7)
                    } catch let error as EmbeddingError where error.kind == .unavailable {}
                    for input in [[], [" "]] as [[String]] {
                        do {
                            _ = try await missing.embed(input, for: .document)
                            exit(8)
                        } catch let error as EmbeddingError where error.kind == .unavailable {}
                    }
                    exit(0)
                }
                _ = await ExitTestTimer.finish("loadAfterShutdown", watchdog)
            }
        }

        /// Lets the parent of an exit test kill a child that hangs,
        /// and measure how long the child takes to end after the step under test.
        ///
        /// Swift Testing's time limit doesn't end a child process,
        /// so a child that hangs would keep the test run waiting.
        /// The child writes its process ID when it starts,
        /// and the monotonic time when the step under test starts,
        /// to files that both processes name from the parent's process ID.
        /// The measurement starts after the child's setup,
        /// so a slow load doesn't change it.
        enum ExitTestTimer {
            /// How long the parent waits for a child before it kills the child.
            static let limit: Duration = .seconds(30)

            /// The current time of a monotonic clock that every process on the system shares.
            static func now() -> UInt64 {
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            }

            private static func file(_ name: String, _ suffix: String, parent: pid_t) -> URL {
                URL(fileURLWithPath: NSTemporaryDirectory())
                    .appendingPathComponent("llama-exit-test-\(parent)-\(name).\(suffix)")
            }

            /// In the child: records its process ID.
            static func childStarted(_ name: String) {
                try? String(getpid()).write(to: file(name, "pid", parent: getppid()), atomically: true, encoding: .utf8)
            }

            /// In the child: records when the step under test starts.
            static func stepStarted(_ name: String) {
                try? String(now()).write(to: file(name, "start", parent: getppid()), atomically: true, encoding: .utf8)
            }

            private static func read(_ name: String, _ suffix: String) -> String? {
                try? String(contentsOf: file(name, suffix, parent: getpid()), encoding: .utf8)
            }

            /// In the parent: starts a task that kills the child if it hasn't ended after ``limit``,
            /// and returns whether it did.
            static func watch(_ name: String) -> Task<Bool, Never> {
                for suffix in ["pid", "start"] {
                    try? FileManager.default.removeItem(at: file(name, suffix, parent: getpid()))
                }
                return Task.detached {
                    guard (try? await Task.sleep(for: limit)) != nil else { return false }
                    guard let pid = read(name, "pid").flatMap({ pid_t($0) }) else { return false }
                    kill(pid, SIGKILL)
                    return true
                }
            }

            /// In the parent, after the child ends: stops the watchdog,
            /// records an issue if it killed the child,
            /// and returns the time from the step's start to now.
            static func finish(_ name: String, _ watchdog: Task<Bool, Never>) async -> Duration? {
                watchdog.cancel()
                let killed = await watchdog.value
                #expect(!killed, "The child process didn't end within \(limit).")
                guard let start = read(name, "start").flatMap({ UInt64($0) }) else { return nil }
                return .nanoseconds(Int64(now() - start))
            }
        }
    #endif

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
