#if Llama
    import Foundation
    import LlamaSwift

    /// An embedding model that runs a GGUF file with llama.cpp.
    ///
    /// The model loads the file on the first embedding or on ``prewarm()``.
    /// llama.cpp pools each text into one vector
    /// with the pooling type in the file's metadata,
    /// and the model scales each vector to unit length.
    /// A file without a pooling type,
    /// such as a model that generates text,
    /// or a reranker, which pools to a score,
    /// fails to load with ``LlamaEmbeddingModelError/notEmbeddingModel``.
    ///
    /// ```swift
    /// let model = LlamaEmbeddingModel(
    ///     modelPath: "/path/to/bge-small-en-v1.5-q8_0.gguf",
    ///     name: "bge-small-en-v1.5-q8_0",
    ///     version: "1",
    ///     dimension: 384,
    ///     prefixes: .bge
    /// )
    /// let vectors = try await model.embed(["I like tea."], for: .document)
    /// ```
    ///
    /// Copies of a model share one loaded copy of its weights.
    ///
    /// ggml's Metal backend stops a process with an assertion
    /// if the process exits while a model is still loaded on the GPU.
    /// The package tries to prevent this in an exit handler,
    /// which stops new work, waits a short time for running work,
    /// and frees the models that it loaded if no work is still running.
    /// The handler can't help in these cases,
    /// and the process can still stop with the assertion:
    ///
    /// - An exit during the first load,
    ///   because the handler is registered when a load ends.
    /// - An exit from inside a llama.cpp call or callback.
    /// - Work that is still running after ``LlamaLifecycle/exitDeadline``.
    /// - An exit that runs no exit handlers, such as `_exit` or a crash.
    ///
    /// To be certain that no model is loaded when the process exits,
    /// finish every load, ``prewarm()``, and embedding,
    /// start no new work, and then call ``removeAllFromCache()``.
    /// The call cancels a running load without waiting for it,
    /// and that load can still finish and cache its model.
    ///
    /// The package can't tell whether a file has changed,
    /// so ``version`` is your promise:
    /// change it whenever you change the file,
    /// or stores will mix vectors from the old and new files.
    public struct LlamaEmbeddingModel: EmbeddingModel {
        /// The reason that the model can't embed text now.
        public enum UnavailableReason: Sendable, Hashable {
            /// The model isn't loaded yet.
            /// The next embedding or ``LlamaEmbeddingModel/prewarm()`` loads it.
            case notLoaded

            /// The last load failed, with a description of the error.
            /// The next embedding tries again.
            case failedToLoad(String)

            /// The process is exiting, and the package has stopped its llama.cpp work.
            /// Every load and embedding throws ``EmbeddingError/Kind/unavailable``.
            case exiting
        }

        /// The verbosity level for llama.cpp logging.
        ///
        /// This type has the same cases as `LlamaLanguageModel.LogLevel`
        /// in AnyLanguageModel.
        public enum LogLevel: Int, Hashable, Comparable, Sendable, CaseIterable {
            /// No logging output.
            case none = 0
            /// Debug messages and above (most verbose).
            case debug = 1
            /// Info messages and above.
            case info = 2
            /// Warning messages and above (default).
            case warn = 3
            /// Only error messages.
            case error = 4

            /// The ggml log level that corresponds to this level.
            var ggmlLevel: ggml_log_level {
                switch self {
                case .none: GGML_LOG_LEVEL_NONE
                case .debug: GGML_LOG_LEVEL_DEBUG
                case .info: GGML_LOG_LEVEL_INFO
                case .warn: GGML_LOG_LEVEL_WARN
                case .error: GGML_LOG_LEVEL_ERROR
                }
            }

            public static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
                lhs.rawValue < rhs.rawValue
            }
        }

        /// The minimum level of the llama.cpp messages that the process writes to standard error.
        ///
        /// llama.cpp has one log callback for the whole process,
        /// so this setting applies to every model.
        /// The package installs its callback once, before it initializes llama.cpp,
        /// because llama.cpp doesn't protect the callback against concurrent changes.
        /// Setting this property changes only the level that the callback reads.
        ///
        /// AnyLanguageModel's `LlamaLanguageModel.logLevel` installs its own callback,
        /// and the callback that was installed last decides which messages appear.
        public static var logLevel: LogLevel {
            get { LlamaRuntime.logLevel.withLock { $0 } }
            set { LlamaRuntime.logLevel.withLock { $0 = newValue } }
        }

        /// The number of layers that a model offloads to the GPU by default.
        ///
        /// A negative value offloads every layer.
        /// The llama.cpp binaries are built with Metal and embed its shader library.
        public static var defaultGPULayerCount: Int32 { -1 }

        /// The path to the GGUF model file.
        public let modelPath: String

        /// A name for the model in ``identifier``.
        ///
        /// The identifier doesn't use the path,
        /// so that moving the file doesn't change it.
        public let name: String

        /// The version that you chose for the file.
        ///
        /// Change it whenever the file changes.
        public let version: String

        /// The number of elements in each vector.
        public let dimension: Int

        /// The text that the model puts before each input.
        ///
        /// GGUF files don't record the prefixes of a model,
        /// so set them to match the model.
        public let prefixes: EmbeddingPrefixes

        /// The maximum number of tokens that the model reads from one text,
        /// including special tokens.
        ///
        /// A longer text is truncated without an error.
        ///
        /// The model also stops at the context length that the model was trained with,
        /// if it is smaller.
        /// The model removes the tokens after the limit
        /// and keeps the last token,
        /// which is a separator or end token in most tokenizers.
        public let maximumTokenCount: Int

        /// The maximum number of texts that the model runs in one call to llama.cpp.
        ///
        /// The model also limits each call to a number of tokens,
        /// which is the larger of ``maximumTokenCount`` and 512,
        /// so a call can hold fewer texts than this number.
        public let maximumBatchSize: Int

        /// The number of model layers to offload to the GPU.
        ///
        /// A negative value offloads all layers, and `0` runs entirely on the CPU.
        /// The GPU and the CPU can give slightly different vectors,
        /// within floating-point error,
        /// so this setting isn't part of ``identifier``.
        public let gpuLayers: Int32

        /// A string that identifies the vectors that this model produces.
        ///
        /// The identifier has ``name``, ``version``, ``dimension``,
        /// and a fingerprint of ``prefixes`` and ``maximumTokenCount``.
        public let identifier: String

        /// The key of the loader that this model shares with its copies.
        let loaderKey: ModelLoaderKey

        private let loader: ModelLoader<LlamaEmbeddingEngine>

        /// The settings of the contexts that this model creates.
        private let contextSettings: LlamaEmbeddingEngine.ContextSettings

        /// Creates a model from a GGUF file.
        ///
        /// - Parameters:
        ///   - modelPath: The path to the GGUF model file.
        ///   - name: A name for the model in ``identifier``.
        ///   - version: A version that you change whenever the file changes.
        ///   - dimension: The number of elements in each vector.
        ///     Loading the model throws ``EmbeddingError/Kind/dimensionMismatch``
        ///     if the model produces another size.
        ///   - prefixes: The text before each query and document.
        ///   - maximumTokenCount: The maximum number of tokens that the model reads from one text.
        ///   - maximumBatchSize: The maximum number of texts that the model runs in one call.
        ///   - gpuLayers: The number of model layers to offload to the GPU.
        ///     Defaults to ``defaultGPULayerCount``.
        public init(
            modelPath: String,
            name: String,
            version: String,
            dimension: Int,
            prefixes: EmbeddingPrefixes = .none,
            maximumTokenCount: Int = 512,
            maximumBatchSize: Int = 32,
            gpuLayers: Int32 = LlamaEmbeddingModel.defaultGPULayerCount
        ) {
            precondition(!name.isEmpty, "The name must not be empty.")
            precondition(!version.isEmpty, "The version must not be empty.")
            precondition(dimension > 0, "The dimension must be greater than zero.")
            precondition(maximumTokenCount > 1, "The token count must be greater than one.")
            precondition(maximumBatchSize > 0, "The batch size must be greater than zero.")
            self.modelPath = modelPath
            self.name = name
            self.version = version
            self.dimension = dimension
            self.prefixes = prefixes
            self.maximumTokenCount = maximumTokenCount
            self.maximumBatchSize = maximumBatchSize
            self.gpuLayers = gpuLayers
            self.identifier = Identifier.make(
                engine: "llama",
                model: name,
                revision: version,
                dimension: dimension,
                settings: [prefixes.query, prefixes.document, String(maximumTokenCount)]
            )
            let settings = LlamaEmbeddingEngine.ContextSettings(
                maximumTokenCount: maximumTokenCount,
                maximumBatchSize: maximumBatchSize,
                threads: Int32(ProcessInfo.processInfo.processorCount)
            )
            self.contextSettings = settings
            // The number of GPU layers is part of the key,
            // because the model applies it when it loads the file.
            let key = ModelLoaderKey(
                source: .file,
                location: modelPath,
                version: "\(version)\0\(gpuLayers)"
            )
            self.loaderKey = key
            self.loader = LlamaLoaderRegistry.loader(for: key) {
                try LlamaLifecycle.shared.perform {
                    try LlamaEmbeddingEngine(path: modelPath, gpuLayers: gpuLayers, settings: settings)
                }
            }
        }

        // MARK: EmbeddingModel

        /// Whether the weights are in memory.
        ///
        /// A model that isn't loaded is still usable:
        /// the next embedding loads it.
        public var availability: Availability<UnavailableReason> {
            if LlamaLifecycle.shared.isStopping {
                return .unavailable(.exiting)
            }
            return switch loader.state {
            case .loaded: .available
            case .failed(let description): .unavailable(.failedToLoad(description))
            case .notLoaded: .unavailable(.notLoaded)
            }
        }

        /// Loads the model, and checks that it is an embedding model with ``dimension`` elements.
        ///
        /// - Throws: ``LlamaEmbeddingModelError`` if the file can't be loaded
        ///   or isn't an embedding model,
        ///   ``EmbeddingError/Kind/dimensionMismatch``
        ///   if the model produces vectors of another size than ``dimension``,
        ///   or ``EmbeddingError/Kind/unavailable`` if the process is exiting.
        public func prewarm() async throws {
            // After the exit handler starts, every request throws the exit error first,
            // even for a missing file or a model that is in the cache.
            try LlamaLifecycle.shared.stopIfExiting()
            let engine = try await loader.container()
            guard engine.dimension == dimension else {
                throw EmbeddingError.dimensionMismatch(expected: dimension, actual: engine.dimension)
            }
        }

        /// Removes this model's weights from the shared cache and cancels any load in progress.
        ///
        /// The removal applies to every copy of this model,
        /// including copies that exist already.
        /// An embedding that is running keeps the weights until it ends.
        /// The next embedding loads the weights again.
        public func removeFromCache() async {
            loader.unload()
        }

        /// Removes the weights of every llama.cpp embedding model from the shared cache
        /// and cancels every load in progress.
        ///
        /// Models that exist already observe the removal,
        /// and their next embedding loads the weights again.
        /// An embedding that is running keeps its model's weights until it ends.
        ///
        /// To be certain that no model is loaded when the process exits,
        /// finish every load, ``prewarm()``, and embedding,
        /// start no new work, and then call this method.
        /// It cancels a running load without waiting for it,
        /// and that load can still finish and cache its model.
        /// The package's exit handler is best effort; see ``LlamaEmbeddingModel``.
        public static func removeAllFromCache() async {
            LlamaLoaderRegistry.unloadAll()
        }

        /// Returns a vector for each text, and loads the model if it isn't loaded.
        ///
        /// The model puts as many texts in each call to llama.cpp
        /// as ``maximumBatchSize`` and the token limit of a call allow.
        ///
        /// After the package's exit handler starts,
        /// this method throws ``EmbeddingError/Kind/unavailable`` for every input,
        /// including an empty array.
        public func embed(_ texts: [String], for purpose: EmbeddingPurpose) async throws -> [[Float]] {
            // After the exit handler starts, every request throws the exit error first,
            // before the checks of its input.
            try LlamaLifecycle.shared.stopIfExiting()
            try Vectors.validate(texts)
            guard !texts.isEmpty else { return [] }
            let engine = try await loader.container()
            try Task.checkCancellation()
            let inputs = texts.map { prefixes.apply(to: $0, for: purpose) }
            let settings = contextSettings
            let vectors = try LlamaLifecycle.shared.perform { try engine.embed(inputs, settings: settings) }
            return try Vectors.normalized(vectors, count: texts.count, dimension: dimension)
        }

        /// Returns the token IDs that the model reads for each text,
        /// with the prefix for `purpose` and after truncation.
        func tokenIDs(for texts: [String], purpose: EmbeddingPurpose) async throws -> [[Int32]] {
            try LlamaLifecycle.shared.stopIfExiting()
            let engine = try await loader.container()
            let limit = engine.tokenLimit(for: contextSettings)
            let inputs = texts.map { prefixes.apply(to: $0, for: purpose) }
            return try LlamaLifecycle.shared.perform {
                try inputs.map { try engine.tokenize($0, limit: limit) }
            }
        }
    }

    /// An error that ``LlamaEmbeddingModel`` throws when llama.cpp can't load or run a model.
    ///
    /// The cases follow `LlamaLanguageModelError` in AnyLanguageModel.
    public enum LlamaEmbeddingModelError: Error, LocalizedError, Hashable, Sendable {
        /// No file exists at the model path.
        case invalidModelPath

        /// llama.cpp couldn't load the file as a model.
        case modelLoadFailed

        /// llama.cpp couldn't create a context for the model.
        case contextInitializationFailed

        /// The model's pooling type doesn't produce one vector for each text.
        ///
        /// The model accepts mean, CLS, and last-token pooling.
        /// A model that generates text has no pooling type,
        /// so llama.cpp returns a vector for each token,
        /// and a reranker uses rank pooling, which returns a score.
        case notEmbeddingModel

        /// The model has an encoder and a decoder, such as T5,
        /// and llama.cpp doesn't compute embeddings for these models.
        case encoderDecoderModel

        /// llama.cpp couldn't tokenize a text.
        case tokenizationFailed

        /// llama.cpp couldn't run a batch, or returned no vector for a text.
        case decodingFailed

        public var errorDescription: String? {
            switch self {
            case .invalidModelPath:
                "Invalid model file path"
            case .modelLoadFailed:
                "Failed to load model from file"
            case .contextInitializationFailed:
                "Failed to initialize context"
            case .notEmbeddingModel:
                "This model doesn't pool each text into one vector, so it isn't an embedding model"
            case .encoderDecoderModel:
                "This model has an encoder and a decoder, and llama.cpp can't embed text with it"
            case .tokenizationFailed:
                "Failed to tokenize input text"
            case .decodingFailed:
                "Failed to compute embeddings"
            }
        }
    }

    // MARK: - Runtime

    /// The state that llama.cpp keeps for the whole process.
    enum LlamaRuntime {
        static let logLevel = Locked(LlamaEmbeddingModel.LogLevel.warn)

        /// The key of a thread-local value that is set
        /// if the callback wrote the last message on the thread that wasn't a continuation.
        ///
        /// llama.cpp writes a message and its continuations from one thread,
        /// and messages from other threads can come in between.
        private static let continuationKey: pthread_key_t = {
            var key = pthread_key_t()
            precondition(pthread_key_create(&key, nil) == 0, "Failed to create a thread-local key.")
            return key
        }()

        /// Returns whether the callback writes a message,
        /// and records the decision for the message's continuations
        /// on the current thread.
        ///
        /// A continuation has no level of its own,
        /// so it follows the decision for the last message on the same thread,
        /// even if the level changed after that message.
        /// At ``LlamaEmbeddingModel/LogLevel/none``, no message is written.
        static func shouldWrite(level: ggml_log_level, minimum: LlamaEmbeddingModel.LogLevel) -> Bool {
            let writes: Bool
            if level == GGML_LOG_LEVEL_CONT {
                writes = pthread_getspecific(continuationKey) != nil
            } else {
                writes = minimum != .none && level.rawValue >= minimum.ggmlLevel.rawValue
                pthread_setspecific(continuationKey, writes ? UnsafeRawPointer(bitPattern: 1) : nil)
            }
            return writes && minimum != .none
        }

        /// Initializes the llama.cpp backend once,
        /// and installs the log callback first,
        /// so that loading the first model logs at the current level.
        ///
        /// This is the only call to `llama_log_set`.
        /// `llama.h` says that the logger state is global
        /// and that its functions aren't thread-safe,
        /// so the callback is installed before any other llama.cpp call
        /// and never replaced.
        static let initialized: Void = {
            llama_log_set(llamaLogCallback, nil)
            llama_backend_init()
        }()

        /// Registers ``LlamaLifecycle/shutDown(deadline:)`` to run when the process exits.
        ///
        /// ggml creates static objects as it needs them,
        /// such as its Metal device and buffer types during a model's first load,
        /// and the C runtime destroys them at exit
        /// in the reverse order of the exit handlers and static objects that came before.
        /// A handler runs before the objects created before its registration are destroyed,
        /// but after the objects created later.
        /// An earlier registration, before the first load, ran after ggml's objects were destroyed
        /// and crashed in `llama_free`.
        ///
        /// ggml doesn't report when it creates such an object,
        /// so the package can't tell whether an earlier registration still runs early enough.
        /// So every load registers the handler again when it ends,
        /// and so does each model's first embedding, which compiles ggml's GPU kernels.
        /// The last registration runs first, and the handler does nothing after its first run.
        ///
        /// If a registration fails,
        /// an earlier one could run after objects that the models use are destroyed,
        /// so the handler then frees nothing.
        static func registerExitHandler() {
            if atexit({ _ = LlamaLifecycle.shared.shutDown() }) != 0 {
                LlamaLifecycle.shared.exitHandlerRegistrationFailed()
            }
        }
    }

    /// Writes llama.cpp messages at or above ``LlamaEmbeddingModel/logLevel`` to standard error.
    ///
    /// A C callback can't capture Swift context,
    /// so this function reads the level from ``LlamaRuntime``.
    /// Besides the write, each message costs one lock and unlock to read the level,
    /// and one or two reads or writes of a thread-local value.
    private func llamaLogCallback(
        level: ggml_log_level,
        text: UnsafePointer<CChar>?,
        userData: UnsafeMutableRawPointer?
    ) {
        #if DEBUG
            LlamaTestHooks.onLog.withLock { $0 }?()
        #endif
        let minimum = LlamaRuntime.logLevel.withLock { $0 }
        guard LlamaRuntime.shouldWrite(level: level, minimum: minimum), let text else { return }
        fputs(String(cString: text), stderr)
    }

    /// The loaders of every llama.cpp model in this process, by the key of the file that they load.
    ///
    /// As with the MLX models,
    /// the registry keeps each loader for as long as the process runs.
    private enum LlamaLoaderRegistry {
        static let loaders = Locked<[ModelLoaderKey: ModelLoader<LlamaEmbeddingEngine>]>([:])

        /// Returns the loader for a key, and creates it with `load` if there is none.
        static func loader(
            for key: ModelLoaderKey,
            load: @escaping @Sendable () async throws -> LlamaEmbeddingEngine
        ) -> ModelLoader<LlamaEmbeddingEngine> {
            loaders.withLock { loaders in
                if let loader = loaders[key] { return loader }
                let loader = ModelLoader(load: load)
                loaders[key] = loader
                return loader
            }
        }

        /// Unloads every loader, which every model that holds one observes.
        static func unloadAll() {
            let all = loaders.withLock { Array($0.values) }
            for loader in all {
                loader.unload()
            }
        }
    }

    // MARK: - Lifecycle

    /// Tracks the llama.cpp work in this process,
    /// and tries to free every model when the process exits.
    ///
    /// ggml's Metal backend frees its device in a C++ static destructor at exit,
    /// and it stops the process with an assertion
    /// if a model or a context still has buffers on the device then.
    /// Plain llama.cpp calls do the same
    /// when a process exits with a model loaded on the GPU.
    /// The loader registry keeps models for as long as the process runs,
    /// so the package tries to free them in an exit handler.
    ///
    /// The handler is best effort.
    /// It must never make an exit worse than an exit without it,
    /// so it never waits without a limit,
    /// and it never frees a model or a context that a running call could use.
    /// It stops new loads and embeddings,
    /// and it waits up to ``exitDeadline`` for the ones that are running:
    /// a load stops at its next progress report,
    /// and an embedding stops between texts while it tokenizes them
    /// and between batches.
    /// If no work is running then,
    /// it frees the model and the cached context of every model that is still loaded,
    /// including models that Swift objects still hold.
    /// Otherwise it frees nothing.
    /// Freeing only the idle models wouldn't help,
    /// because the busy model's buffers would still cause the assertion.
    ///
    /// The handler can't help, and the assertion can still happen,
    /// in these cases:
    ///
    /// - An exit during the first load or its first context,
    ///   because the handler is registered when a load ends.
    ///   An earlier registration would run after ggml's objects are destroyed.
    /// - An exit from inside a llama.cpp call or callback.
    ///   The handler sees that its thread is running an operation
    ///   and returns without waiting.
    /// - Work that is still running at the deadline.
    /// - An exit that runs no exit handlers, such as `_exit` or a crash.
    ///
    /// After the handler starts, every load and embedding throws
    /// ``EmbeddingError/Kind/unavailable``.
    final class LlamaLifecycle: @unchecked Sendable {
        static let shared = LlamaLifecycle()

        /// How long the exit handler waits for running loads and embeddings.
        ///
        /// A running embedding stops between texts while it tokenizes them
        /// and between batches,
        /// so the wait is about one call to llama.cpp:
        /// one batch of at most ``LlamaEmbeddingModel/maximumBatchSize`` texts,
        /// which takes milliseconds for the test model.
        /// Three seconds leaves room for one batch of a much larger model,
        /// and still ends a process that can't finish its work
        /// soon enough that the delay looks like a slow quit, not a hang.
        static let exitDeadline: TimeInterval = 3

        /// How long the exit handler's cleanup may take before the process aborts.
        ///
        /// The cleanup frees every model after running work has ended.
        /// llama.cpp calls the installed log callback while it frees,
        /// and another library can replace that callback with one that blocks,
        /// for example on a lock that the exiting thread holds,
        /// and a GPU driver can stall while it releases buffers.
        /// Without a limit the process would then hang,
        /// where without the handler it would have stopped with ggml's assertion.
        /// So the cleanup runs under a watchdog,
        /// which calls `abort()` if the cleanup is still running after this time.
        /// It tries to write one line to standard error first,
        /// but it aborts whether or not the line can be written.
        ///
        /// Freeing the test model, `bge-small-en-v1.5-q8_0.gguf` (37 MB), and its context
        /// took 0.3 to 0.7 milliseconds on an Apple silicon Mac, on the GPU and on the CPU.
        /// AnyLanguageModel measured about 35 milliseconds for a 2.5 GB model.
        /// The work grows with the number and size of the buffers,
        /// so even an embedding model of tens of gigabytes would take well under a second.
        /// Ten seconds, the same limit as AnyLanguageModel's,
        /// leaves a wide margin for a slow but working free.
        static let cleanupDeadline: TimeInterval = 10

        /// What ``shutDown(deadline:)`` did.
        enum ShutdownResult: Equatable {
            /// It freed every model.
            case freed
            /// It had already run, so it did nothing.
            case alreadyShutDown
            /// It ran on a thread inside an operation, so it freed nothing without waiting.
            case insideOperation
            /// Work was still running at the deadline, so it freed nothing.
            case timedOut
            /// A registration of the exit handler failed, so it freed nothing.
            case registrationFailed
        }

        /// The key in a thread's dictionary for the number of operations that the thread runs.
        private static let depthKey = "AnyEmbeddingModel.LlamaLifecycle.operationDepth"

        /// Guards the properties below, and signals when no operation is running.
        private let condition = NSCondition()
        private var isShuttingDown = false
        private var registrationFailed = false
        private var activeOperations = 0
        private var resources: [ObjectIdentifier: LlamaModelResources] = [:]

        /// Whether the process is exiting.
        var isStopping: Bool {
            condition.withLock { isShuttingDown }
        }

        /// The number of loads and embeddings that are running.
        var activeOperationCount: Int {
            condition.withLock { activeOperations }
        }

        /// Runs a load or an embedding,
        /// unless the process is exiting.
        func perform<Result>(_ body: () throws -> Result) throws -> Result {
            try condition.withLock {
                guard !isShuttingDown else { throw Self.exiting }
                activeOperations += 1
            }
            return try runCounted(body)
        }

        /// Runs `body` as work that the caller has already counted,
        /// marks the current thread as running it,
        /// and stops counting it when it returns.
        private func runCounted<Result>(_ body: () throws -> Result) rethrows -> Result {
            // The body is synchronous, so it stays on this thread.
            let thread = Thread.current.threadDictionary
            let depth = thread[Self.depthKey] as? Int ?? 0
            thread[Self.depthKey] = depth + 1
            defer {
                thread[Self.depthKey] = depth
                condition.withLock {
                    activeOperations -= 1
                    if activeOperations == 0 { condition.broadcast() }
                }
            }
            return try body()
        }

        /// Throws ``exiting`` if the exit handler has started.
        func stopIfExiting() throws(EmbeddingError) {
            if isStopping { throw Self.exiting }
        }

        /// Whether the current thread is running an operation.
        private var isInsideOperation: Bool {
            (Thread.current.threadDictionary[Self.depthKey] as? Int ?? 0) > 0
        }

        /// Records that a registration of the exit handler failed.
        func exitHandlerRegistrationFailed() {
            condition.withLock { registrationFailed = true }
        }

        /// The error that work throws after the exit handler starts.
        static var exiting: EmbeddingError {
            .unavailable("The process is exiting, so the package has stopped its llama.cpp work.")
        }

        /// Records the resources of a model, so that ``shutDown()`` can free them.
        func track(_ model: LlamaModelResources) {
            condition.withLock { resources[ObjectIdentifier(model)] = model }
        }

        /// Frees the resources of a model, unless ``shutDown(deadline:)`` freed them already.
        ///
        /// The cleanup runs even while the process is exiting.
        /// It counts as running work,
        /// so that the exit handler's thread check and deadline apply to it,
        /// and it calls llama.cpp without the lock:
        /// llama.cpp logs while it frees a model,
        /// and a log callback that calls `exit` runs the exit handler on this thread,
        /// which takes the lock.
        func free(_ model: LlamaModelResources) {
            condition.withLock {
                resources[ObjectIdentifier(model)] = nil
                activeOperations += 1
            }
            runCounted { model.free() }
        }

        /// Stops new work, waits up to `deadline` for the work that is running,
        /// and frees the resources of every model if no work is running then.
        @discardableResult
        func shutDown(deadline: TimeInterval = exitDeadline) -> ShutdownResult {
            let insideOperation = isInsideOperation
            var models: [LlamaModelResources] = []
            let result: ShutdownResult = condition.withLock {
                guard !isShuttingDown else { return .alreadyShutDown }
                isShuttingDown = true
                // The operation on this thread can't end while its thread waits here.
                guard !insideOperation else { return .insideOperation }
                guard !registrationFailed else { return .registrationFailed }
                // Measure the deadline with a monotonic clock,
                // and wait in short steps that start from the current wall-clock time,
                // so that a change to the wall clock can't extend the wait.
                let clock = ContinuousClock()
                let end = clock.now + .seconds(deadline)
                while activeOperations > 0 {
                    let remaining = clock.now.duration(to: end)
                    guard remaining > .zero else { break }
                    let step = min(remaining, .milliseconds(100))
                    _ = condition.wait(until: Date(timeIntervalSinceNow: step / .seconds(1)))
                }
                guard activeOperations == 0 else { return .timedOut }
                models = Array(resources.values)
                resources = [:]
                return .freed
            }
            // Free without the lock, as in `free(_:)`.
            // If a log callback calls `exit` here, this handler runs again and does nothing.
            // With nothing to free, no watchdog starts.
            guard !models.isEmpty else { return result }
            Self.withAbortDeadline(Self.cleanupDeadline) {
                for model in models {
                    model.free()
                }
            }
            return result
        }

        /// The state of a cleanup that runs under ``withAbortDeadline(_:_:)``.
        private enum CleanupState: Sendable {
            case running
            case completed
            case aborting
        }

        /// Runs `body`, and aborts the process if `body` is still running after `deadline`.
        ///
        /// A separate thread measures the time on a monotonic clock,
        /// so the exiting thread can't stop it.
        /// The thread ends as soon as `body` returns,
        /// and doesn't keep the process alive.
        ///
        /// Completion and the timeout are decided in one step under a lock.
        /// Once `body` has returned and completion is recorded,
        /// the watchdog never aborts.
        /// Once the watchdog has recorded the timeout,
        /// this function aborts instead of returning,
        /// even if `body` returns after that.
        static func withAbortDeadline<Result>(_ deadline: TimeInterval, _ body: () -> Result) -> Result {
            let state = Locked(CleanupState.running)
            let finished = DispatchSemaphore(value: 0)
            let watchdog = Thread {
                guard finished.wait(timeout: .now() + deadline) == .timedOut else { return }
                #if DEBUG
                    LlamaTestHooks.cleanupWatchdog.withLock { $0 }?(.beforeDecision)
                #endif
                let aborts = state.withLock { state in
                    guard state == .running else { return false }
                    state = .aborting
                    return true
                }
                #if DEBUG
                    LlamaTestHooks.cleanupWatchdog.withLock { $0 }?(.decided(aborts: aborts))
                #endif
                if aborts {
                    abortAfterExitCleanupTimeout()
                }
            }
            watchdog.start()
            let result = body()
            let completed = state.withLock { state in
                guard state == .running else { return false }
                state = .completed
                return true
            }
            // The watchdog has decided to abort, and is about to.
            guard completed else { abortAfterLateExitCleanup() }
            finished.signal()
            return result
        }

        /// Aborts the process because the exit handler's cleanup returned
        /// after the watchdog had decided to abort.
        ///
        /// A crash report for this abort shows this function
        /// in the backtrace of the thread that called `abort()`.
        /// It writes no message and doesn't wait for the watchdog.
        @inline(never)
        private static func abortAfterLateExitCleanup() -> Never {
            abort()
        }

        /// Aborts the process because the exit handler's cleanup took too long.
        ///
        /// A crash report for this abort shows this function
        /// in the backtrace of the thread that called `abort()`,
        /// which identifies it.
        /// The function also tries to write one line to standard error,
        /// from another thread and with `write(2)`,
        /// but waits at most 100 milliseconds for it:
        /// standard error can be a full pipe that nobody reads,
        /// and its lock can be held by the blocked thread,
        /// so the abort must not depend on the message.
        @inline(never)
        private static func abortAfterExitCleanupTimeout() -> Never {
            let written = DispatchSemaphore(value: 0)
            let writer = Thread {
                let message = "AnyEmbeddingModel: freeing llama.cpp models at exit didn't finish in time; aborting.\n"
                _ = message.withCString { write(STDERR_FILENO, $0, strlen($0)) }
                written.signal()
            }
            writer.start()
            _ = written.wait(timeout: .now() + .milliseconds(100))
            abort()
        }
    }

    #if DEBUG
        /// Functions that tests run at points inside llama.cpp work.
        ///
        /// Only debug builds, which `swift test` uses, compile these hooks,
        /// so they cost nothing in a release build.
        enum LlamaTestHooks {
            /// Runs at the start of each batch of an embedding,
            /// before its checks for cancellation and exit,
            /// on the embedding's thread.
            ///
            /// After the hook returns,
            /// the embedding does no llama.cpp work before those checks,
            /// so an embedding that the hook holds until the exit handler starts
            /// stops at once.
            static let beforeBatch = Locked<(@Sendable () -> Void)?>(nil)

            /// Runs for each llama.cpp log message, before the level filter,
            /// on the thread that logs it.
            static let onLog = Locked<(@Sendable () -> Void)?>(nil)

            /// A point in the cleanup watchdog after its deadline passes.
            enum CleanupWatchdogEvent: Sendable, Equatable {
                /// The watchdog is about to decide between completion and the timeout.
                case beforeDecision
                /// The watchdog has decided, and aborts if `aborts` is true.
                case decided(aborts: Bool)
            }

            /// Runs on the cleanup watchdog's thread after its deadline passes.
            static let cleanupWatchdog = Locked<(@Sendable (CleanupWatchdogEvent) -> Void)?>(nil)
        }
    #endif

    /// A loaded llama.cpp model and the context that its embeddings reuse.
    ///
    /// ``LlamaLifecycle`` holds these resources until it frees them,
    /// separately from the ``LlamaEmbeddingEngine`` that uses them,
    /// so that the exit handler can free a model that a Swift object still holds.
    final class LlamaModelResources: @unchecked Sendable {
        /// The settings of a context, which depend on the model that creates it.
        struct ContextSettings: Hashable, Sendable {
            var maximumTokenCount: Int
            var maximumBatchSize: Int
            var threads: Int32
        }

        private struct CachedContext {
            let context: OpaquePointer
            let settings: ContextSettings
            var isCheckedOut: Bool
        }

        let model: OpaquePointer
        let vocab: OpaquePointer

        /// Guards `cachedContext` and `isFreed`.
        private let lock = NSLock()
        private var cachedContext: CachedContext?
        private var isFreed = false

        init(model: OpaquePointer, vocab: OpaquePointer) {
            self.model = model
            self.vocab = vocab
        }

        /// Frees the cached context and the model, once.
        ///
        /// Call it only through ``LlamaLifecycle``,
        /// when no load or embedding uses the model.
        func free() {
            // Take the context under the lock, and call llama.cpp without it.
            let (isFirst, context): (Bool, OpaquePointer?) = lock.withLock {
                guard !isFreed else { return (false, nil) }
                isFreed = true
                defer { cachedContext = nil }
                return (true, cachedContext?.context)
            }
            guard isFirst else { return }
            if let context {
                llama_free(context)
            }
            llama_model_free(model)
        }

        /// Puts a new context in the cache.
        func cache(_ context: OpaquePointer, settings: ContextSettings) {
            lock.withLock {
                cachedContext = CachedContext(context: context, settings: settings, isCheckedOut: false)
            }
        }

        /// Returns the cached context if it is free and has `settings`.
        func checkOutCachedContext(_ settings: ContextSettings) -> OpaquePointer? {
            lock.withLock {
                guard var entry = cachedContext, !entry.isCheckedOut, entry.settings == settings else {
                    return nil
                }
                entry.isCheckedOut = true
                cachedContext = entry
                return entry.context
            }
        }

        /// Returns a context to the cache,
        /// or frees it if it isn't the cached context and the cache already has one.
        func checkIn(_ context: OpaquePointer, settings: ContextSettings) {
            // Choose the context to free under the lock, and call llama.cpp without it.
            let unused: OpaquePointer? = lock.withLock {
                if cachedContext?.context == context {
                    cachedContext?.isCheckedOut = false
                    return nil
                }
                let previous = cachedContext
                if let previous {
                    // Keep the context of the most recent settings,
                    // unless another embedding is using the cached one.
                    guard !previous.isCheckedOut, previous.settings != settings else { return context }
                }
                cachedContext = CachedContext(context: context, settings: settings, isCheckedOut: false)
                return previous?.context
            }
            if let unused {
                llama_free(unused)
            }
        }
    }

    // MARK: - Engine

    /// Runs embeddings on a loaded llama.cpp model.
    ///
    /// The model's weights are read-only,
    /// so several contexts can run on them at the same time.
    /// A context isn't safe to share,
    /// so each embedding checks out the cached context for its whole run.
    /// An embedding that finds the context checked out
    /// runs on a context of its own and frees it at the end,
    /// as `LlamaLanguageModel` does with its session context.
    ///
    /// Call the engine's methods, and create it,
    /// only inside ``LlamaLifecycle/perform(_:)``.
    final class LlamaEmbeddingEngine: @unchecked Sendable {
        typealias ContextSettings = LlamaModelResources.ContextSettings

        private let resources: LlamaModelResources
        private var model: OpaquePointer { resources.model }
        private var vocab: OpaquePointer { resources.vocab }

        /// The number of elements in each vector that the model produces.
        let dimension: Int

        /// The context length that the model was trained with, or `nil` if the file doesn't say.
        let trainedContextLength: Int?

        /// Whether an embedding has run, which registers the exit handler again.
        private let hasEmbedded = Locked(false)

        /// Loads the model and checks that it produces one vector for each text.
        init(path: String, gpuLayers: Int32, settings: ContextSettings) throws {
            guard FileManager.default.fileExists(atPath: path) else {
                throw LlamaEmbeddingModelError.invalidModelPath
            }
            LlamaRuntime.initialized

            var params = llama_model_default_params()
            params.n_gpu_layers = gpuLayers
            params.load_mode = LLAMA_LOAD_MODE_MMAP
            // Returning false stops the load, so that the exit handler doesn't wait for all of it.
            params.progress_callback = { _, _ in !LlamaLifecycle.shared.isStopping }
            guard let model = llama_model_load_from_file(path, params) else {
                if LlamaLifecycle.shared.isStopping { throw LlamaLifecycle.exiting }
                throw LlamaEmbeddingModelError.modelLoadFailed
            }
            guard let vocab = llama_model_get_vocab(model) else {
                llama_model_free(model)
                throw LlamaEmbeddingModelError.modelLoadFailed
            }
            let resources = LlamaModelResources(model: model, vocab: vocab)
            LlamaLifecycle.shared.track(resources)
            self.resources = resources
            self.dimension = Int(llama_model_n_embd_out(model))
            let trained = Int(llama_model_n_ctx_train(model))
            self.trainedContextLength = trained > 0 ? trained : nil

            // From here on, deinit frees the model.
            // llama.cpp's embedding example rejects encoder-decoder models.
            if llama_model_has_encoder(model) && llama_model_has_decoder(model) {
                throw LlamaEmbeddingModelError.encoderDecoderModel
            }
            // The exit handler waits for this load, so stop it before the context.
            try LlamaLifecycle.shared.stopIfExiting()
            // The pooling type comes from the file,
            // and a context reports it, so check it with the first context.
            let context = try makeContext(settings)
            switch llama_pooling_type(context) {
            case LLAMA_POOLING_TYPE_MEAN, LLAMA_POOLING_TYPE_CLS, LLAMA_POOLING_TYPE_LAST:
                resources.cache(context, settings: settings)
                LlamaRuntime.registerExitHandler()
            default:
                llama_free(context)
                throw LlamaEmbeddingModelError.notEmbeddingModel
            }
        }

        deinit {
            LlamaLifecycle.shared.free(resources)
        }

        /// The number of tokens that the model reads from one text.
        func tokenLimit(for settings: ContextSettings) -> Int {
            min(settings.maximumTokenCount, trainedContextLength ?? .max)
        }

        /// The number of tokens in one call to llama.cpp.
        ///
        /// A model with non-causal attention, such as BERT,
        /// needs every token of a call in one physical batch,
        /// so the context, the batch, and the physical batch all have this size.
        /// The minimum of 512 lets short texts share a call
        /// when the token limit is small.
        private func tokenCapacity(for settings: ContextSettings) -> Int {
            max(tokenLimit(for: settings), 512)
        }

        /// The number of texts in one call to llama.cpp.
        private func sequenceCapacity(for settings: ContextSettings) -> Int {
            min(settings.maximumBatchSize, llama_max_parallel_sequences())
        }

        private func makeContext(_ settings: ContextSettings) throws -> OpaquePointer {
            let tokens = UInt32(tokenCapacity(for: settings))
            var params = llama_context_default_params()
            params.embeddings = true
            params.n_ctx = tokens
            params.n_batch = tokens
            params.n_ubatch = tokens
            params.n_seq_max = UInt32(sequenceCapacity(for: settings))
            // Every sequence in a call shares one cache of the context's size,
            // as in llama.cpp's embedding example,
            // rather than a slice of it for each sequence.
            params.kv_unified = true
            params.n_threads = settings.threads
            params.n_threads_batch = settings.threads
            guard let context = llama_init_from_model(model, params) else {
                throw LlamaEmbeddingModelError.contextInitializationFailed
            }
            return context
        }

        /// Returns the tokens of `text`, with the model's special tokens, truncated to `limit`.
        func tokenize(_ text: String, limit: Int) throws -> [llama_token] {
            // llama.cpp takes the length of the text and the capacity as 32-bit integers.
            let byteCount = text.utf8.count
            guard byteCount <= Int(Int32.max) else {
                throw LlamaEmbeddingModelError.tokenizationFailed
            }
            let count = Int32(byteCount)
            // Add in `Int`, which can't overflow here, and clamp the result.
            var capacity = max(Int32(clamping: byteCount + 8), 16)
            for _ in 0 ..< 2 {
                var tokens = [llama_token](repeating: 0, count: Int(capacity))
                // llama.cpp's embedding example and server
                // add special tokens and parse special tokens in the text.
                let written = llama_tokenize(vocab, text, count, &tokens, capacity, true, true)
                if written > 0 {
                    return Vectors.truncated(Array(tokens.prefix(Int(written))), limit: limit)
                }
                // A negative result is the number of tokens that the text needs.
                // `Int32.min` has no positive counterpart, so it is an error.
                guard written < 0, written != .min else { break }
                capacity = -written
            }
            throw LlamaEmbeddingModelError.tokenizationFailed
        }

        /// Returns the pooled vector of each input, in input order, without normalization.
        func embed(_ inputs: [String], settings: ContextSettings) throws -> [[Float]] {
            let limit = tokenLimit(for: settings)
            // The exit handler waits for this embedding, so stop it at each step.
            var tokens: [[llama_token]] = []
            tokens.reserveCapacity(inputs.count)
            for input in inputs {
                try LlamaLifecycle.shared.stopIfExiting()
                tokens.append(try tokenize(input, limit: limit))
            }
            try LlamaLifecycle.shared.stopIfExiting()
            let context = try resources.checkOutCachedContext(settings) ?? makeContext(settings)
            defer { resources.checkIn(context, settings: settings) }

            let capacity = tokenCapacity(for: settings)
            let batch = llama_batch_init(Int32(capacity), 0, 1)
            defer { llama_batch_free(batch) }

            let groups = Self.groups(
                tokens.map(\.count),
                tokenCapacity: capacity,
                sequenceCapacity: sequenceCapacity(for: settings)
            )
            var vectors: [[Float]] = []
            vectors.reserveCapacity(inputs.count)
            for group in groups {
                #if DEBUG
                    LlamaTestHooks.beforeBatch.withLock { $0 }?()
                #endif
                try Task.checkCancellation()
                try LlamaLifecycle.shared.stopIfExiting()
                vectors += try run(group.map { tokens[$0] }, batch: batch, context: context)
            }
            let isFirstEmbedding = hasEmbedded.withLock { hasEmbedded in
                defer { hasEmbedded = true }
                return !hasEmbedded
            }
            if isFirstEmbedding {
                LlamaRuntime.registerExitHandler()
            }
            return vectors
        }

        /// Splits texts into consecutive groups,
        /// each with at most `tokenCapacity` tokens and `sequenceCapacity` texts.
        ///
        /// - Parameter counts: The number of tokens in each text.
        ///   No count can be greater than `tokenCapacity`.
        /// - Returns: The indices of the texts in each group.
        static func groups(_ counts: [Int], tokenCapacity: Int, sequenceCapacity: Int) -> [Range<Int>] {
            var groups: [Range<Int>] = []
            var start = 0
            var tokens = 0
            for (index, count) in counts.enumerated() {
                precondition(count <= tokenCapacity, "A text has more tokens than a call can hold.")
                if index > start && (tokens + count > tokenCapacity || index - start == sequenceCapacity) {
                    groups.append(start ..< index)
                    start = index
                    tokens = 0
                }
                tokens += count
            }
            if start < counts.count {
                groups.append(start ..< counts.count)
            }
            return groups
        }

        /// Runs one call with one sequence for each text, and returns the pooled vectors.
        private func run(
            _ sequences: [[llama_token]],
            batch: llama_batch,
            context: OpaquePointer
        ) throws -> [[Float]] {
            var batch = batch
            batch.n_tokens = 0
            for (sequence, tokens) in sequences.enumerated() {
                for (position, token) in tokens.enumerated() {
                    let index = Int(batch.n_tokens)
                    batch.token[index] = token
                    batch.pos[index] = llama_pos(position)
                    batch.n_seq_id[index] = 1
                    batch.seq_id[index]![0] = llama_seq_id(sequence)
                    batch.logits[index] = 1
                    batch.n_tokens += 1
                }
            }
            // A model with a cache, such as a decoder, would otherwise
            // attend to the previous call's tokens.
            if let memory = llama_get_memory(context) {
                llama_memory_clear(memory, true)
            }
            // llama_decode runs the encoder for a model without a cache, such as BERT.
            guard llama_decode(context, batch) == 0 else {
                throw LlamaEmbeddingModelError.decodingFailed
            }
            return try sequences.indices.map { sequence in
                guard let vector = llama_get_embeddings_seq(context, llama_seq_id(sequence)) else {
                    throw LlamaEmbeddingModelError.decodingFailed
                }
                return Array(UnsafeBufferPointer(start: vector, count: dimension))
            }
        }
    }
#endif
