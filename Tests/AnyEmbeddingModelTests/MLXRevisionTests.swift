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
      // The server's base URL ends in /v1, which a Hub host doesn't have.
      let hub = HubClient(host: url.deletingLastPathComponent(), cache: nil)
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
          modelId: "org/model", dimension: 3, hub: hub)
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
          modelId: "org/model", revision: "stable", dimension: 3, hub: hub)
        let identifier = before.identifier

        server.setBehavior(.json(modelJSON(sha: commitB)))
        let after = try await MLXEmbeddingModel.resolving(
          modelId: "org/model", revision: "stable", dimension: 3, hub: hub)

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
          modelId: "org/model", dimension: 3, prefixes: .bge, hub: hub)
        let requests = server.requests.count
        let pinned = try MLXEmbeddingModel(
          modelId: "org/model", commit: resolved.revision, dimension: 3, prefixes: .bge)
        #expect(pinned.identifier == resolved.identifier)
        #expect(server.requests.count == requests)
      }
    }

    @Test("Resolving a commit hash makes no request")
    func resolvingACommit() async throws {
      try await withHub(.json("{}")) { server, hub in
        let model = try await MLXEmbeddingModel.resolving(
          modelId: "org/model", revision: commitA, dimension: 3, hub: hub)
        #expect(model.revision == commitA)
        #expect(server.requests.isEmpty)
      }
    }

    @Test("A response with no commit is an error")
    func noCommit() async throws {
      try await withHub(.json(#"{"id": "org/model"}"#)) { _, hub in
        await #expect(throws: EmbeddingError.self) {
          try await MLXEmbeddingModel.resolving(modelId: "org/model", dimension: 3, hub: hub)
        }
      }
    }

    @Test("A repository without a namespace is rejected before any request")
    func badRepository() async throws {
      try await withHub(.json("{}")) { server, hub in
        await #expect(throws: EmbeddingError.self) {
          try await MLXEmbeddingModel.resolving(modelId: "model", dimension: 3, hub: hub)
        }
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
      .needsRealModels)
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

      try copyModelFiles(from: try await bge.modelDirectory(), to: directory)
      let first = MLXEmbeddingModel(directory: directory, name: "m", version: "1", dimension: 384)
      defer { first.unload() }
      let firstVector = try await first.embed(text, for: .document)
      #expect(cosine(firstVector, bgeVector) > 0.9999)

      for name in try manager.contentsOfDirectory(atPath: directory.path) {
        try manager.removeItem(at: directory.appendingPathComponent(name))
      }
      try copyModelFiles(from: try await miniLM.modelDirectory(), to: directory)

      let second = MLXEmbeddingModel(directory: directory, name: "m", version: "2", dimension: 384)
      defer { second.unload() }
      #expect(second.identifier != first.identifier)
      let secondVector = try await second.embed(text, for: .document)
      #expect(cosine(secondVector, miniLMVector) > 0.9999)
      #expect(cosine(secondVector, firstVector) < 0.9)
    }

    @Test("A local model's identifier uses its version and not its path")
    func localModel() {
      let here = MLXEmbeddingModel(
        directory: URL(fileURLWithPath: "/tmp/one"), name: "m", version: "1", dimension: 3)
      let moved = MLXEmbeddingModel(
        directory: URL(fileURLWithPath: "/tmp/two"), name: "m", version: "1", dimension: 3)
      let newer = MLXEmbeddingModel(
        directory: URL(fileURLWithPath: "/tmp/one"), name: "m", version: "2", dimension: 3)
      #expect(here.identifier == moved.identifier)
      #expect(here.identifier != newer.identifier)
    }
  }
#endif
