#if MLX
    import Foundation
    import HuggingFace
    import MLX
    import MLXEmbedders
    import MLXHuggingFace
    import MLXLMCommon
    import Tokenizers

    /// An embedding model that runs on the GPU with MLX.
    ///
    /// The model downloads its weights and tokenizer from the Hugging Face Hub
    /// on the first embedding or on ``prewarm()``,
    /// or loads them from a local directory.
    /// The tests compare each preset with sentence-transformers at a pinned commit,
    /// for both purposes, for batches of unequal texts, and at the truncation limit.
    /// The `bgeSmallEnglish`, `allMiniLM`, and `nomicEmbedText` presets agree to a cosine similarity
    /// of at least 0.99999.
    /// `qwen3Embedding(task:)` is quantized to 4 bits, and it describes its own agreement.
    /// `MLXEmbedders` can load other architectures,
    /// but this engine pools their hidden states itself,
    /// and those results haven't been checked against a reference.
    ///
    /// Use a preset, such as ``bgeSmallEnglish``,
    /// to get the dimension and prefixes of a known model:
    ///
    /// ```swift
    /// let model = MLXEmbeddingModel.bgeSmallEnglish
    /// try await model.prewarm()
    /// let vectors = try await model.embed(["I like tea."], for: .document)
    /// ```
    ///
    /// Copies of a model share one loaded copy of its weights.
    ///
    /// ## Revisions
    ///
    /// A branch such as `main` can move to new weights
    /// while its name stays the same.
    /// A store that saves vectors with ``identifier``
    /// would then mix old vectors with new ones and never know.
    /// So every model has an exact version before it exists:
    ///
    /// - The presets, such as ``bgeSmallEnglish``, pin a commit.
    /// - ``init(modelId:commit:dimension:prefixes:maximumTokenCount:maximumBatchSize:hub:)``
    ///   takes a full commit hash and throws for anything else.
    /// - ``resolving(modelId:revision:dimension:prefixes:maximumTokenCount:maximumBatchSize:hub:)``
    ///   asks the Hub which commit a branch or tag names now,
    ///   and returns a model pinned to that commit.
    /// - ``init(directory:name:version:dimension:prefixes:maximumTokenCount:maximumBatchSize:)``
    ///   takes a version that you choose for files on disk.
    public struct MLXEmbeddingModel: EmbeddingModel {
        /// The reason that the model can't embed text now.
        public enum UnavailableReason: Sendable, Hashable {
            /// The model isn't loaded yet.
            /// The next embedding or ``MLXEmbeddingModel/prewarm()`` loads it.
            case notLoaded

            /// The last load failed, with a description of the error.
            /// The next embedding tries again.
            case failedToLoad(String)
        }

        /// The repository of the model on the Hugging Face Hub,
        /// such as `"BAAI/bge-small-en-v1.5"`.
        public let modelId: String

        /// The exact version of the model:
        /// a full commit hash for a model from the Hub,
        /// or the version that you chose for a model in a local directory.
        ///
        /// After ``resolving(modelId:revision:dimension:prefixes:maximumTokenCount:maximumBatchSize:hub:)``,
        /// save this value and pass it to
        /// ``init(modelId:commit:dimension:prefixes:maximumTokenCount:maximumBatchSize:hub:)``
        /// to skip the branch lookup next time.
        /// Loading the model still requires its files.
        public let revision: String

        /// A local directory with the model files,
        /// or `nil` to download them.
        public let directory: URL?

        /// The number of elements in each vector.
        public let dimension: Int

        /// The text that the model puts before each input.
        public let prefixes: EmbeddingPrefixes

        /// The maximum number of tokens that the model reads from one text,
        /// including special tokens.
        ///
        /// A longer text is truncated without an error.
        ///
        /// The model also stops at the architecture's own limit, if it is smaller.
        /// The model removes the tokens after the limit
        /// and keeps the last token,
        /// which is a separator or end token in most tokenizers.
        public let maximumTokenCount: Int

        /// The maximum number of texts that the model runs at one time.
        ///
        /// A larger batch uses more memory, and padding makes short texts wait for long ones.
        public let maximumBatchSize: Int

        /// A string that identifies the vectors that this model produces.
        ///
        /// The identifier has ``modelId``, ``revision``, ``dimension``,
        /// and a fingerprint of ``prefixes`` and ``maximumTokenCount``.
        public let identifier: String

        /// The key of the loader that this model shares with its copies.
        let loaderKey: ModelLoaderKey

        private let loader: ModelLoader<EmbedderModelContainer>

        /// Creates a model from a commit of a Hugging Face Hub repository.
        ///
        /// - Parameters:
        ///   - modelId: The repository of the model on the Hugging Face Hub.
        ///   - commit: The full 40-character commit hash.
        ///     To start from a branch or tag name,
        ///     call ``resolving(modelId:revision:dimension:prefixes:maximumTokenCount:maximumBatchSize:hub:)``.
        ///   - dimension: The number of elements in each vector.
        ///     The first embedding throws ``EmbeddingError/dimensionMismatch(expected:actual:)``
        ///     if the model produces another size.
        ///   - prefixes: The text before each query and document.
        ///   - maximumTokenCount: The maximum number of tokens that the model reads from one text.
        ///   - maximumBatchSize: The maximum number of texts that the model runs at one time.
        ///   - hub: The client that downloads the model, or `nil` for the default client.
        /// - Throws: ``EmbeddingError/invalidCommit(_:)``
        ///   if `commit` isn't a full commit hash.
        public init(
            modelId: String,
            commit: String,
            dimension: Int,
            prefixes: EmbeddingPrefixes = .none,
            maximumTokenCount: Int = 512,
            maximumBatchSize: Int = 32,
            hub: HubClient? = nil
        ) throws(EmbeddingError) {
            guard Self.isCommitHash(commit) else { throw .invalidCommit(commit) }
            self.init(
                pinned: modelId,
                commit: commit,
                dimension: dimension,
                prefixes: prefixes,
                maximumTokenCount: maximumTokenCount,
                maximumBatchSize: maximumBatchSize,
                hub: hub
            )
        }

        /// Creates a model from files in a local directory.
        ///
        /// The package can't tell whether the files in a directory have changed,
        /// so `version` is your promise:
        /// change it whenever you change the files,
        /// or stores will mix vectors from the old and new files.
        /// Each version loads its files once,
        /// so don't change the files of a version while it is in use.
        ///
        /// - Parameters:
        ///   - directory: A directory with the weights, the configuration, and the tokenizer.
        ///   - name: A name for the model in ``identifier``.
        ///     The identifier doesn't use the path,
        ///     so that moving the directory doesn't change it.
        ///   - version: A version that you change whenever the files change.
        ///   - dimension: The number of elements in each vector.
        ///   - prefixes: The text before each query and document.
        ///   - maximumTokenCount: The maximum number of tokens that the model reads from one text.
        ///   - maximumBatchSize: The maximum number of texts that the model runs at one time.
        public init(
            directory: URL,
            name: String,
            version: String,
            dimension: Int,
            prefixes: EmbeddingPrefixes = .none,
            maximumTokenCount: Int = 512,
            maximumBatchSize: Int = 32
        ) {
            precondition(!name.isEmpty, "The name must not be empty.")
            precondition(!version.isEmpty, "The version must not be empty.")
            self.init(
                modelId: name,
                revision: version,
                directory: directory,
                dimension: dimension,
                prefixes: prefixes,
                maximumTokenCount: maximumTokenCount,
                maximumBatchSize: maximumBatchSize,
                hub: nil
            )
        }

        /// Asks the Hub which commit a branch or tag names now,
        /// and returns a model pinned to that commit.
        ///
        /// This is the only call that follows a branch.
        /// The model keeps its commit for as long as it exists,
        /// so its ``identifier`` can't change under a store.
        /// If the branch moves,
        /// the next call returns a model with a different identifier,
        /// and a store embeds its memories again.
        ///
        /// This call needs the network unless `revision` is already a full commit hash.
        /// Save ``revision`` from the result and pass it to
        /// ``init(modelId:commit:dimension:prefixes:maximumTokenCount:maximumBatchSize:hub:)``
        /// to skip this lookup and keep the same vectors.
        /// Loading the model still requires its files.
        ///
        /// - Parameters:
        ///   - modelId: The repository of the model on the Hugging Face Hub.
        ///   - revision: A branch, a tag, or a commit.
        ///   - dimension: The number of elements in each vector.
        ///   - prefixes: The text before each query and document.
        ///   - maximumTokenCount: The maximum number of tokens that the model reads from one text.
        ///   - maximumBatchSize: The maximum number of texts that the model runs at one time.
        ///   - hub: The client that asks the Hub and downloads the model,
        ///     or `nil` for the default client.
        /// - Returns: A model pinned to the commit.
        /// - Throws: ``EmbeddingError/invalidResponse(_:)`` if the Hub reports no commit,
        ///   ``EmbeddingError/unavailable(_:)`` if `modelId` isn't of the form `namespace/name`,
        ///   or an error from the Hub client if the request fails.
        public static func resolving(
            modelId: String,
            revision: String = "main",
            dimension: Int,
            prefixes: EmbeddingPrefixes = .none,
            maximumTokenCount: Int = 512,
            maximumBatchSize: Int = 32,
            hub: HubClient? = nil
        ) async throws -> MLXEmbeddingModel {
            let commit: String
            if isCommitHash(revision) {
                commit = revision
            } else {
                guard let repository = Repo.ID(rawValue: modelId) else {
                    throw EmbeddingError.unavailable("\(modelId) isn't of the form namespace/name.")
                }
                let model = try await (hub ?? HubClient()).getModel(repository, revision: revision)
                guard let sha = model.sha, isCommitHash(sha) else {
                    throw EmbeddingError.invalidResponse(
                        "The Hub reported no commit for \(modelId) at \(revision)."
                    )
                }
                commit = sha
            }
            return MLXEmbeddingModel(
                pinned: modelId,
                commit: commit,
                dimension: dimension,
                prefixes: prefixes,
                maximumTokenCount: maximumTokenCount,
                maximumBatchSize: maximumBatchSize,
                hub: hub
            )
        }

        /// Returns whether `string` is a full, lowercase, 40-character commit hash.
        static func isCommitHash(_ string: String) -> Bool {
            string.utf8.count == 40
                && string.utf8.allSatisfy { (48 ... 57).contains($0) || (97 ... 102).contains($0) }
        }

        /// Creates a model from a commit that is already known to be valid.
        private init(
            pinned modelId: String,
            commit: String,
            dimension: Int,
            prefixes: EmbeddingPrefixes = .none,
            maximumTokenCount: Int = 512,
            maximumBatchSize: Int = 32,
            hub: HubClient? = nil
        ) {
            self.init(
                modelId: modelId,
                revision: commit,
                directory: nil,
                dimension: dimension,
                prefixes: prefixes,
                maximumTokenCount: maximumTokenCount,
                maximumBatchSize: maximumBatchSize,
                hub: hub
            )
        }

        private init(
            modelId: String,
            revision: String,
            directory: URL?,
            dimension: Int,
            prefixes: EmbeddingPrefixes,
            maximumTokenCount: Int,
            maximumBatchSize: Int,
            hub: HubClient?
        ) {
            precondition(dimension > 0, "The dimension must be greater than zero.")
            precondition(maximumTokenCount > 1, "The token count must be greater than one.")
            precondition(maximumBatchSize > 0, "The batch size must be greater than zero.")
            self.modelId = modelId
            self.revision = revision
            self.directory = directory
            self.dimension = dimension
            self.prefixes = prefixes
            self.maximumTokenCount = maximumTokenCount
            self.maximumBatchSize = maximumBatchSize
            self.identifier = Identifier.make(
                engine: "mlx",
                model: modelId,
                revision: revision,
                dimension: dimension,
                settings: [prefixes.query, prefixes.document, String(maximumTokenCount), "pool-v2"]
            )
            // A local version is part of the key,
            // so that a new version of a directory loads its new files.
            let key = ModelLoaderKey(
                source: directory == nil ? .hub : .directory,
                location: directory?.path ?? modelId,
                version: revision
            )
            self.loaderKey = key
            self.loader = LoaderRegistry.loader(for: key) {
                let tokenizerLoader = #huggingFaceTokenizerLoader()
                if let directory {
                    return try await EmbedderModelFactory.shared.loadContainer(
                        from: directory,
                        using: tokenizerLoader
                    )
                }
                let downloader = if let hub { #hubDownloader(hub) } else { #hubDownloader() }
                return try await EmbedderModelFactory.shared.loadContainer(
                    from: downloader,
                    using: tokenizerLoader,
                    configuration: ModelConfiguration(id: modelId, revision: revision)
                )
            }
        }

        // MARK: Presets

        /// BAAI's `bge-small-en-v1.5`: 384 dimensions, CLS pooling, with a query instruction.
        public static var bgeSmallEnglish: MLXEmbeddingModel {
            MLXEmbeddingModel(
                pinned: "BAAI/bge-small-en-v1.5",
                commit: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
                dimension: 384,
                prefixes: .bge
            )
        }

        /// Sentence Transformers' `all-MiniLM-L6-v2`: 384 dimensions, mean pooling, no prefixes.
        public static var allMiniLM: MLXEmbeddingModel {
            MLXEmbeddingModel(
                pinned: "sentence-transformers/all-MiniLM-L6-v2",
                commit: "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                dimension: 384,
                maximumTokenCount: 256
            )
        }

        /// Nomic's `nomic-embed-text-v1.5`: 768 dimensions, mean pooling,
        /// with a prefix for each purpose.
        public static var nomicEmbedText: MLXEmbeddingModel {
            MLXEmbeddingModel(
                pinned: "nomic-ai/nomic-embed-text-v1.5",
                commit: "e9b6763023c676ca8431644204f50c2b100d9aab",
                dimension: 768,
                prefixes: .nomic
            )
        }

        /// Qwen's `Qwen3-Embedding-0.6B`, quantized to 4 bits by the MLX community:
        /// 1,024 dimensions, last-token pooling, with an instruction before each query.
        ///
        /// Quantization changes the vectors.
        /// Compared with the full-precision model in sentence-transformers,
        /// the cosine similarity is at least 0.955 for ordinary sentences,
        /// and as low as 0.70 for a text of one word repeated up to the token limit.
        /// Don't mix these vectors with vectors from the full-precision model;
        /// the two models have different identifiers.
        ///
        /// - Parameter task: The task description in the query instruction.
        public static func qwen3Embedding(
            task: String = "Given a web search query, retrieve relevant passages that answer the query"
        ) -> MLXEmbeddingModel {
            MLXEmbeddingModel(
                pinned: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
                commit: "6c3ae70858513f1a78e9cdca3cae330d9075cd2a",
                dimension: 1024,
                prefixes: .qwen3(task: task)
            )
        }

        // MARK: EmbeddingModel

        /// Whether the weights are in memory.
        ///
        /// A model that isn't loaded is still usable:
        /// the next embedding loads it.
        public var availability: Availability<UnavailableReason> {
            switch loader.state {
            case .loaded: .available
            case .failed(let description): .unavailable(.failedToLoad(description))
            case .notLoaded: .unavailable(.notLoaded)
            }
        }

        /// Downloads and loads the model, and checks its ``dimension``.
        ///
        /// - Throws: ``EmbeddingError/dimensionMismatch(expected:actual:)``
        ///   if the model produces vectors of another size than ``dimension``.
        public func prewarm() async throws {
            _ = try await embed(["dimension check"], for: .document)
        }

        /// Returns the local directory of the model files,
        /// and downloads and loads the model if it isn't loaded.
        public func modelDirectory() async throws -> URL {
            try await loader.container().modelDirectory
        }

        /// Releases the weights of this model and of every copy of it.
        ///
        /// The next embedding loads them again.
        public func unload() {
            loader.unload()
        }

        /// Returns a vector for each text, and loads the model if it isn't loaded.
        ///
        /// A batch larger than ``maximumBatchSize`` runs in several groups, one after another.
        public func embed(_ texts: [String], for purpose: EmbeddingPurpose) async throws -> [[Float]] {
            try Vectors.validate(texts)
            guard !texts.isEmpty else { return [] }
            let container = try await loader.container()
            var vectors: [[Float]] = []
            vectors.reserveCapacity(texts.count)
            for batch in Vectors.batches(texts, size: maximumBatchSize) {
                try Task.checkCancellation()
                let inputs = batch.map { prefixes.apply(to: $0, for: purpose) }
                let limit = maximumTokenCount
                let batchVectors = try await container.perform { context in
                    try Self.run(inputs, maximumTokenCount: limit, context: context)
                }
                vectors += try Vectors.normalized(
                    batchVectors,
                    count: batch.count,
                    dimension: dimension,
                    firstIndex: batch.startIndex
                )
            }
            return vectors
        }

        /// Returns the token IDs that the model reads for each text,
        /// with the prefix for `purpose` and after truncation.
        func tokenIDs(
            for texts: [String],
            purpose: EmbeddingPurpose
        ) async throws -> [[Int]] {
            let container = try await loader.container()
            let inputs = texts.map { prefixes.apply(to: $0, for: purpose) }
            let maximumTokenCount = maximumTokenCount
            return await container.perform { context in
                let limit = min(maximumTokenCount, context.model.maxPositionEmbeddings ?? .max)
                return Self.tokenIDs(inputs, limit: limit, tokenizer: context.tokenizer)
            }
        }

        private static func tokenIDs(
            _ inputs: [String],
            limit: Int,
            tokenizer: any MLXLMCommon.Tokenizer
        ) -> [[Int]] {
            inputs.map { truncated(tokenizer.encode(text: $0, addSpecialTokens: true), limit: limit) }
        }

        /// Returns `ids` with at most `limit` tokens.
        ///
        /// The last token stays, because the tokenizer's template puts a special token there:
        /// a separator for BERT, and `<|endoftext|>` for Qwen3, whose pooling reads it.
        /// That token can differ from the tokenizer's `eos_token`,
        /// so this function never adds one.
        static func truncated(_ ids: [Int], limit: Int) -> [Int] {
            guard ids.count > limit, let last = ids.last else { return ids }
            return Array(ids.prefix(limit - 1)) + [last]
        }

        /// Tokenizes, pads, and runs one batch, and pools the hidden states.
        private static func run(
            _ inputs: [String],
            maximumTokenCount: Int,
            context: EmbedderModelContext
        ) throws -> [[Float]] {
            let tokenizer = context.tokenizer
            let strategy = context.pooling.strategy
            let limit = min(maximumTokenCount, context.model.maxPositionEmbeddings ?? .max)
            let tokens = tokenIDs(inputs, limit: limit, tokenizer: tokenizer)
            let length = tokens.map(\.count).max() ?? 0
            let padding = tokenizer.eosTokenId ?? 0
            var ids: [Int32] = []
            var mask: [Int32] = []
            ids.reserveCapacity(tokens.count * length)
            mask.reserveCapacity(tokens.count * length)
            for row in tokens {
                ids += row.map(Int32.init) + Array(repeating: Int32(padding), count: length - row.count)
                mask +=
                    Array(repeating: 1, count: row.count) + Array(repeating: 0, count: length - row.count)
            }
            let shape = [tokens.count, length]
            let inputIds = MLXArray(ids, shape)
            let attentionMask = MLXArray(mask, shape)
            let output = context.model(
                inputIds,
                positionIds: nil,
                tokenTypeIds: MLXArray.zeros(like: inputIds),
                attentionMask: attentionMask
            )
            guard let hidden = output.hiddenStates else {
                throw EmbeddingError.invalidResponse("The model returned no hidden states.")
            }
            let pooled = pool(hidden, mask: attentionMask, strategy: strategy)
            pooled.eval()
            return (0 ..< tokens.count).map { pooled[$0].asArray(Float.self) }
        }

        /// Pools hidden states of shape `[batch, length, dimension]` into `[batch, dimension]`.
        ///
        /// `MLXEmbedders.Pooling` uses a BERT model's pooler output for CLS pooling,
        /// which applies a dense layer and `tanh` to the first token.
        /// Sentence Transformers uses the first token's hidden state,
        /// so this function does the same.
        private static func pool(
            _ hidden: MLXArray,
            mask: MLXArray,
            strategy: Pooling.Strategy
        ) -> MLXArray {
            let weights = mask.asType(hidden.dtype).expandedDimensions(axes: [-1])
            switch strategy {
            case .mean:
                return sum(hidden * weights, axis: 1) / sum(weights, axis: 1)
            case .max:
                // Padding must never win, so give it the lowest value.
                let masked = which(weights .> 0, hidden, MLXArray(-Float.greatestFiniteMagnitude))
                return MLX.max(masked, axis: 1)
            case .last:
                let counts = sum(mask.asType(.int32), axis: -1)
                let indices = MLX.maximum(counts - 1, 0).expandedDimensions(axes: [1, 2])
                return takeAlong(hidden, indices, axis: 1).squeezed(axis: 1)
            case .cls, .first, .none:
                return hidden[0..., 0, 0...]
            }
        }
    }

    /// The loaders of every model in this process, by the key of the files that they load.
    private enum LoaderRegistry {
        static let loaders = Locked<[ModelLoaderKey: ModelLoader<EmbedderModelContainer>]>([:])

        /// Returns the loader for a key, and creates it with `load` if there is none.
        static func loader(
            for key: ModelLoaderKey,
            load: @escaping @Sendable () async throws -> EmbedderModelContainer
        ) -> ModelLoader<EmbedderModelContainer> {
            loaders.withLock { loaders in
                if let loader = loaders[key] { return loader }
                let loader = ModelLoader(load: load)
                loaders[key] = loader
                return loader
            }
        }
    }
#endif
