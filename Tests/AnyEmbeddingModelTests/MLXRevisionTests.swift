#if MLX
    import Foundation
    import HuggingFace
    import Testing

    @testable import AnyEmbeddingModel

    /// These tests use a local server for the Hub and need no network.
    @Suite("MLXEmbeddingModel revisions", .serialized)
    struct MLXRevisionTests {
        private let commitA = String(repeating: "a", count: 40)
        private let commitB = String(repeating: "b", count: 39) + "0"

        private func withHub(
            _ behavior: MockEmbeddingServer.Behavior,
            _ body: (MockEmbeddingServer, HubClient) async throws -> Void
        ) async throws {
            let server = try MockEmbeddingServer(behavior: behavior)
            let url = try await server.start()
            defer { server.stop() }
            // A download needs a cache, so each client has an empty one of its own.
            let manager = FileManager.default
            let cache = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try manager.createDirectory(at: cache, withIntermediateDirectories: true)
            defer { try? manager.removeItem(at: cache) }
            // The server's base URL ends in /v1, which a Hub host doesn't have.
            let hub = HubClient(host: url.deletingLastPathComponent(), cache: HubCache(cacheDirectory: cache))
            try await body(server, hub)
        }

        private func modelJSON(sha: String) -> String {
            #"{"id": "org/model", "sha": "\#(sha)"}"#
        }

        @Test("A commit hash becomes part of the identifier")
        func commitInIdentifier() throws {
            let model = try MLXEmbeddingModel(modelId: "org/model", commit: commitA, dimension: 3)
            #expect(model.revision == commitA)
            #expect(model.identifier.hasPrefix("mlx/org%2Fmodel/\(commitA)/3/"))
            #expect(model.availability == .unavailable(.notLoaded))
        }

        @Test(
            "Anything but a full commit hash is rejected",
            arguments: ["main", "v1.0", "", "abc123", String(repeating: "A", count: 40)]
        )
        func rejectsBranches(revision: String) {
            #expect(throws: EmbeddingError.invalidCommit(revision)) {
                try MLXEmbeddingModel(modelId: "org/model", commit: revision, dimension: 3)
            }
        }

        /// The expected values come from `mise run generate:goldens`.
        @Test("Identifiers match the independent implementation")
        func identifierGoldens() {
            #expect(
                MLXEmbeddingModel.bgeSmallEnglish.identifier
                    == "mlx/BAAI%2Fbge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/384/17ad23889b01fd7d"
            )
            #expect(
                MLXEmbeddingModel.allMiniLM.identifier
                    == "mlx/sentence-transformers%2Fall-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/384/e51e86d76f482409"
            )
            #expect(
                MLXEmbeddingModel.nomicEmbedText.identifier
                    == "mlx/nomic-ai%2Fnomic-embed-text-v1.5/e9b6763023c676ca8431644204f50c2b100d9aab/768/b01ebd07491df7f0"
            )
            #expect(
                MLXEmbeddingModel.qwen3Embedding().identifier
                    == "mlx/mlx-community%2FQwen3-Embedding-0.6B-4bit-DWQ/6c3ae70858513f1a78e9cdca3cae330d9075cd2a/1024/ffbd1da9e469cda1"
            )
            #expect(
                MLXEmbeddingModel(
                    directory: URL(fileURLWithPath: "/tmp/golden"),
                    name: "my model",
                    version: "v/1",
                    dimension: 3
                ).identifier == "mlx/my%20model/v%2F1/3/d681d181ef369d3a"
            )
        }

        @Test("Each preset pins a full commit hash")
        func presetsArePinned() {
            let presets: [MLXEmbeddingModel] = [
                .bgeSmallEnglish, .allMiniLM, .nomicEmbedText, .qwen3Embedding(),
            ]
            for preset in presets {
                #expect(MLXEmbeddingModel.isCommitHash(preset.revision), "\(preset.modelId)")
                #expect(preset.identifier.contains(preset.revision))
            }
        }

        /// This test uses the network.
        @Test("Each preset's commit exists on the Hub", .needsRealModels)
        func presetsExistOnTheHub() async throws {
            let presets: [MLXEmbeddingModel] = [
                .bgeSmallEnglish, .allMiniLM, .nomicEmbedText, .qwen3Embedding(),
            ]
            let hub = HubClient()
            for preset in presets {
                let repository = try #require(Repo.ID(rawValue: preset.modelId))
                let model = try await hub.getModel(repository, revision: preset.revision)
                #expect(model.sha == preset.revision, "\(preset.modelId)")

                let resolved = try await MLXEmbeddingModel.resolving(
                    modelId: preset.modelId,
                    dimension: preset.dimension,
                    prefixes: preset.prefixes,
                    maximumTokenCount: preset.maximumTokenCount
                )
                // The presets pin the commit at `main` when they were last checked.
                #expect(MLXEmbeddingModel.isCommitHash(resolved.revision))
            }
        }

        @Test("Resolving a branch asks the Hub and pins the commit")
        func resolvingPins() async throws {
            try await withHub(.json(modelJSON(sha: commitA))) { server, hub in
                let model = try await MLXEmbeddingModel.resolving(
                    modelId: "org/model",
                    dimension: 3,
                    hub: hub
                )
                #expect(model.revision == commitA)
                // The Hub client ends the URL with an empty query.
                let paths = server.requests.map { $0.path.split(separator: "?").first.map(String.init) }
                #expect(paths == ["/api/models/org/model/revision/main"])
            }
        }

        @Test("When a branch moves, the next resolution has a new identifier")
        func branchMoves() async throws {
            try await withHub(.json(modelJSON(sha: commitA))) { server, hub in
                let before = try await MLXEmbeddingModel.resolving(
                    modelId: "org/model",
                    revision: "stable",
                    dimension: 3,
                    hub: hub
                )
                let identifier = before.identifier

                server.setBehavior(.json(modelJSON(sha: commitB)))
                let after = try await MLXEmbeddingModel.resolving(
                    modelId: "org/model",
                    revision: "stable",
                    dimension: 3,
                    hub: hub
                )

                #expect(after.revision == commitB)
                #expect(after.identifier != identifier)
                // The earlier value keeps its commit, so its vectors can't change.
                #expect(before.identifier == identifier)
            }
        }

        @Test("A resolved commit loads the same identifier without the Hub")
        func resolvedCommitRoundTrips() async throws {
            try await withHub(.json(modelJSON(sha: commitA))) { server, hub in
                let resolved = try await MLXEmbeddingModel.resolving(
                    modelId: "org/model",
                    dimension: 3,
                    prefixes: .bge,
                    hub: hub
                )
                let requests = server.requests.count
                let pinned = try MLXEmbeddingModel(
                    modelId: "org/model",
                    commit: resolved.revision,
                    dimension: 3,
                    prefixes: .bge
                )
                #expect(pinned.identifier == resolved.identifier)
                #expect(server.requests.count == requests)
            }
        }

        @Test("A model keeps the Hub client that it was given")
        func hubProperty() async throws {
            try await withHub(.json(modelJSON(sha: commitA))) { _, hub in
                let pinned = try MLXEmbeddingModel(modelId: "org/model", commit: commitA, dimension: 3, hub: hub)
                #expect(pinned.hub === hub)
                let resolved = try await MLXEmbeddingModel.resolving(modelId: "org/model", dimension: 3, hub: hub)
                #expect(resolved.hub === hub)
            }
            #expect(MLXEmbeddingModel.bgeSmallEnglish.hub == nil)
            let local = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/m"),
                name: "m",
                version: "1",
                dimension: 3
            )
            #expect(local.hub == nil)
        }

        @Test("Models with different Hub clients don't share a loader")
        func hubClientKeys() throws {
            let host = URL(string: "http://127.0.0.1:9")!
            let first = HubClient(host: host, cache: nil)
            let second = HubClient(host: host, cache: nil)
            let a = try MLXEmbeddingModel(modelId: "org/model", commit: commitA, dimension: 3, hub: first)
            let b = try MLXEmbeddingModel(modelId: "org/model", commit: commitA, dimension: 3, hub: second)
            let c = try MLXEmbeddingModel(modelId: "org/model", commit: commitA, dimension: 3, hub: first)
            let d = try MLXEmbeddingModel(modelId: "org/model", commit: commitA, dimension: 3)
            #expect(a.loaderKey != b.loaderKey)
            #expect(a.loaderKey == c.loaderKey)
            #expect(a.loaderKey != d.loaderKey)
            #expect(a.identifier == b.identifier)
        }

        @Test("A model with a working client retries with it after another client failed", .needsMetalLibrary)
        func hubClientRetry() async throws {
            try await withHub(.text(status: 503, body: "down")) { failing, failingHub in
                try await withHub(.text(status: 404, body: "missing")) { working, workingHub in
                    let commit = String(repeating: "c", count: 40)
                    let a = try MLXEmbeddingModel(modelId: "org/model", commit: commit, dimension: 3, hub: failingHub)
                    let b = try MLXEmbeddingModel(modelId: "org/model", commit: commit, dimension: 3, hub: workingHub)
                    try await removingFromCache(a, b) {
                        await #expect(throws: (any Error).self) { try await a.prewarm() }
                        #expect(isFailed(a))
                        let failingRequests = failing.requests.count
                        #expect(failingRequests > 0)

                        // The replacement model uses its own client, not the one that failed.
                        #expect(b.availability == .unavailable(.notLoaded))
                        await #expect(throws: (any Error).self) { try await b.prewarm() }
                        #expect(!working.requests.isEmpty)
                        #expect(failing.requests.count == failingRequests)

                        // A retry of the first model uses its own client again.
                        await #expect(throws: (any Error).self) { try await a.prewarm() }
                        #expect(failing.requests.count > failingRequests)
                    }
                }
            }
        }

        @Test("Resolving a commit hash makes no request")
        func resolvingACommit() async throws {
            try await withHub(.json("{}")) { server, hub in
                let model = try await MLXEmbeddingModel.resolving(
                    modelId: "org/model",
                    revision: commitA,
                    dimension: 3,
                    hub: hub
                )
                #expect(model.revision == commitA)
                #expect(server.requests.isEmpty)
            }
        }

        @Test("A response with no commit is an error")
        func noCommit() async throws {
            try await withHub(.json(#"{"id": "org/model"}"#)) { _, hub in
                let error = await #expect(throws: EmbeddingError.self) {
                    try await MLXEmbeddingModel.resolving(modelId: "org/model", dimension: 3, hub: hub)
                }
                #expect(error?.kind == .invalidResponse)
            }
        }

        @Test("A repository without a namespace is rejected before any request")
        func badRepository() async throws {
            try await withHub(.json("{}")) { server, hub in
                let error = await #expect(throws: EmbeddingError.self) {
                    try await MLXEmbeddingModel.resolving(modelId: "model", dimension: 3, hub: hub)
                }
                #expect(error?.kind == .unavailable)
                #expect(server.requests.isEmpty)
            }
        }

        /// Copies the files that MLX reads from a model directory, following the cache's links.
        private func copyModelFiles(from source: URL, to destination: URL) throws {
            let manager = FileManager.default
            for name in try manager.contentsOfDirectory(atPath: source.path)
            where name.hasSuffix(".json") || name.hasSuffix(".safetensors") || name.hasSuffix(".txt") {
                try manager.copyItem(
                    at: source.appendingPathComponent(name).resolvingSymlinksInPath(),
                    to: destination.appendingPathComponent(name)
                )
            }
            let pooling = source.appendingPathComponent("1_Pooling/config.json")
            if manager.fileExists(atPath: pooling.path) {
                let directory = destination.appendingPathComponent("1_Pooling")
                try manager.createDirectory(at: directory, withIntermediateDirectories: true)
                try manager.copyItem(
                    at: pooling.resolvingSymlinksInPath(),
                    to: directory.appendingPathComponent("config.json")
                )
            }
        }

        @Test(
            "A new version of a local directory loads the files that replaced the old ones",
            .needsRealModels
        )
        func replacedWeightsReload() async throws {
            let manager = FileManager.default
            let directory = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? manager.removeItem(at: directory) }

            // Both models have 384 dimensions, so only the weights tell them apart.
            let bge = MLXEmbeddingModel.bgeSmallEnglish
            let miniLM = MLXEmbeddingModel.allMiniLM
            let text = "My sister lives in Lisbon."
            let bgeVector = try await bge.embed(text, for: .document)
            let miniLMVector = try await miniLM.embed(text, for: .document)
            #expect(cosine(bgeVector, miniLMVector) < 0.9)

            let first = MLXEmbeddingModel(directory: directory, name: "m", version: "1", dimension: 384)
            let second = MLXEmbeddingModel(directory: directory, name: "m", version: "2", dimension: 384)
            try await removingFromCache(first, second) {
                try copyModelFiles(from: try await bge.modelDirectory(), to: directory)
                let firstVector = try await first.embed(text, for: .document)
                #expect(cosine(firstVector, bgeVector) > 0.9999)

                for name in try manager.contentsOfDirectory(atPath: directory.path) {
                    try manager.removeItem(at: directory.appendingPathComponent(name))
                }
                try copyModelFiles(from: try await miniLM.modelDirectory(), to: directory)

                #expect(second.identifier != first.identifier)
                let secondVector = try await second.embed(text, for: .document)
                #expect(cosine(secondVector, miniLMVector) > 0.9999)
                #expect(cosine(secondVector, firstVector) < 0.9)
            }
        }

        @Test("Text in a path and a version can't make two local models share a loader")
        func loaderKeyBoundaries() {
            let a = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/a@b"),
                name: "m",
                version: "c",
                dimension: 3
            )
            let b = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/a"),
                name: "m",
                version: "b@c",
                dimension: 3
            )
            #expect(a.loaderKey != b.loaderKey)
        }

        @Test("Loading one model doesn't change a model whose path and version look alike", .needsMetalLibrary)
        func loaderStateIsSeparate() async throws {
            let base = "/nonexistent-\(UUID().uuidString)/a"
            let a = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: base + "@b"),
                name: "m",
                version: "c",
                dimension: 3
            )
            let b = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: base),
                name: "m",
                version: "b@c",
                dimension: 3
            )
            try await removingFromCache(a, b) {
                await #expect(throws: (any Error).self) { try await a.prewarm() }
                guard case .unavailable(.failedToLoad) = a.availability else {
                    Issue.record("The failed load wasn't reported: \(a.availability)")
                    return
                }
                #expect(b.availability == .unavailable(.notLoaded))

                await b.removeFromCache()
                guard case .unavailable(.failedToLoad) = a.availability else {
                    Issue.record("Removing the other model changed this one: \(a.availability)")
                    return
                }
            }
        }

        /// Returns a model for a directory that doesn't exist, whose loads fail without MLX.
        private func missingDirectoryModel() -> MLXEmbeddingModel {
            MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)"),
                name: "m",
                version: "1",
                dimension: 3
            )
        }

        private func isFailed(_ model: MLXEmbeddingModel) -> Bool {
            if case .unavailable(.failedToLoad) = model.availability { return true }
            return false
        }

        @Test("Removing a model from the cache applies to copies that already exist", .needsMetalLibrary)
        func removeFromCacheReachesCopies() async throws {
            let model = missingDirectoryModel()
            let copy = model
            let twin = MLXEmbeddingModel(
                directory: try #require(model.directory),
                name: "m",
                version: "1",
                dimension: 3
            )
            await #expect(throws: (any Error).self) { try await model.prewarm() }
            #expect(isFailed(copy))
            #expect(isFailed(twin))

            await twin.removeFromCache()
            #expect(model.availability == .unavailable(.notLoaded))
            #expect(copy.availability == .unavailable(.notLoaded))
        }

        @Test("Removing every model from the cache applies to models that already exist", .needsMetalLibrary)
        func removeAllFromCacheReachesModels() async throws {
            let first = missingDirectoryModel()
            let second = missingDirectoryModel()
            await #expect(throws: (any Error).self) { try await first.prewarm() }
            await #expect(throws: (any Error).self) { try await second.prewarm() }
            #expect(isFailed(first))
            #expect(isFailed(second))

            await MLXEmbeddingModel.removeAllFromCache()
            #expect(first.availability == .unavailable(.notLoaded))
            #expect(second.availability == .unavailable(.notLoaded))
        }

        @Test("Canonically equivalent versions and repositories have different loaders")
        func loaderKeyBytes() throws {
            let composed = "caf\u{E9}"
            let decomposed = "cafe\u{301}"
            #expect(composed == decomposed)

            let directory = URL(fileURLWithPath: "/tmp/model")
            let a = MLXEmbeddingModel(directory: directory, name: "m", version: composed, dimension: 3)
            let b = MLXEmbeddingModel(directory: directory, name: "m", version: decomposed, dimension: 3)
            #expect(a.identifier != b.identifier)
            #expect(a.loaderKey != b.loaderKey)

            // A file URL stores its path in one form,
            // so both paths name the same directory and share a loader.
            let c = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/\(composed)"),
                name: "m",
                version: "1",
                dimension: 3
            )
            let d = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/\(decomposed)"),
                name: "m",
                version: "1",
                dimension: 3
            )
            #expect(c.directory?.path == d.directory?.path)
            #expect(c.loaderKey == d.loaderKey)

            // A repository name is text, so its forms are different repositories.

            let e = try MLXEmbeddingModel(modelId: "org/\(composed)", commit: commitA, dimension: 3)
            let f = try MLXEmbeddingModel(modelId: "org/\(decomposed)", commit: commitA, dimension: 3)
            #expect(e.identifier != f.identifier)
            #expect(e.loaderKey != f.loaderKey)
        }

        @Test("A failed load doesn't change a model whose version is canonically equivalent", .needsMetalLibrary)
        func loaderStateByBytes() async throws {
            let directory = URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)")
            let composed = MLXEmbeddingModel(directory: directory, name: "m", version: "\u{E9}", dimension: 3)
            let decomposed = MLXEmbeddingModel(directory: directory, name: "m", version: "e\u{301}", dimension: 3)
            try await removingFromCache(composed, decomposed) {
                await #expect(throws: (any Error).self) { try await composed.prewarm() }
                #expect(isFailed(composed))
                #expect(decomposed.availability == .unavailable(.notLoaded))
            }
        }

        @Test("A directory and a hub model that look alike have different loaders")
        func directoryAndHubModelKeys() throws {
            let path = "/tmp/org/model"
            let directory = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: path),
                name: "m",
                version: commitA,
                dimension: 3
            )
            let hub = try MLXEmbeddingModel(modelId: path, commit: commitA, dimension: 3)
            #expect(directory.loaderKey.location == hub.loaderKey.location)
            #expect(directory.loaderKey.version == hub.loaderKey.version)
            #expect(directory.loaderKey != hub.loaderKey)
        }

        @Test("GPU memory presets have the documented limits, and don't change the identifier")
        func gpuMemoryPresets() {
            let automatic = MLXEmbeddingModel.GPUMemoryConfiguration.automatic
            #expect(automatic.idleCacheLimit == 50_000_000)
            #expect([128_000_000, 256_000_000, 512_000_000, 768_000_000].contains(automatic.activeCacheLimit))
            #expect(automatic.clearCacheOnEviction)
            let unconstrained = MLXEmbeddingModel.GPUMemoryConfiguration.unconstrained
            #expect(unconstrained.activeCacheLimit == .max)
            #expect(unconstrained.idleCacheLimit == .max)
            #expect(!unconstrained.clearCacheOnEviction)

            let directory = URL(fileURLWithPath: "/tmp/gpu")
            let a = MLXEmbeddingModel(directory: directory, name: "m", version: "1", dimension: 3)
            let b = MLXEmbeddingModel(
                directory: directory,
                name: "m",
                version: "1",
                dimension: 3,
                gpuMemory: unconstrained
            )
            #expect(a.gpuMemory == automatic)
            #expect(b.gpuMemory == unconstrained)
            #expect(a.identifier == b.identifier)
        }

        @Test("A load that throws restores the idle cache limit", .needsMetalLibrary)
        func gpuMemoryAfterThrow() async throws {
            let configuration = MLXEmbeddingModel.GPUMemoryConfiguration(
                activeCacheLimit: 123_456_789,
                idleCacheLimit: 98_765_432
            )
            let model = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)"),
                name: "m",
                version: "1",
                dimension: 3,
                gpuMemory: configuration
            )
            try await removingFromCache(model) {
                await #expect(throws: (any Error).self) { try await model.prewarm() }
            }
            let manager = GPUMemoryManager.shared
            #expect(manager.activeScopeCount == 0)
            // The idle limit is the largest of every configuration that the process has used.
            #expect((manager.appliedLimit ?? 0) >= configuration.idleCacheLimit)
            #expect(manager.appliedLimit != configuration.activeCacheLimit)
        }

        @Test("Cancelled and overlapping calls leave no scope active", .needsMetalLibrary)
        func gpuMemoryOverlapping() async throws {
            let models = (0 ..< 8).map { _ in
                MLXEmbeddingModel(
                    directory: URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)"),
                    name: "m",
                    version: "1",
                    dimension: 3
                )
            }
            await withTaskGroup(of: Void.self) { group in
                for (index, model) in models.enumerated() {
                    group.addTask {
                        let task = Task { try await model.prewarm() }
                        if index.isMultiple(of: 2) { task.cancel() }
                        _ = await task.result
                    }
                }
            }
            for model in models { await model.removeFromCache() }
            let manager = GPUMemoryManager.shared
            #expect(manager.activeScopeCount == 0)
            #expect(manager.appliedLimit != nil)
        }

        @Test("A local model's identifier uses its version and not its path")
        func localModel() {
            let here = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/one"),
                name: "m",
                version: "1",
                dimension: 3
            )
            let moved = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/two"),
                name: "m",
                version: "1",
                dimension: 3
            )
            let newer = MLXEmbeddingModel(
                directory: URL(fileURLWithPath: "/tmp/one"),
                name: "m",
                version: "2",
                dimension: 3
            )
            #expect(here.identifier == moved.identifier)
            #expect(here.identifier != newer.identifier)
        }
    }
#endif
