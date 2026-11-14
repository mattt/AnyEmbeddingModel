#if MLX
  import Foundation
  import Testing

  @testable import AnyEmbeddingModel

  /// What sentence-transformers produced for one text.
  private struct ReferenceCase: Decodable, Sendable {
    let name: String
    let purpose: String
    let text: String
    let tokens: [Int]
    let vector: [Float]

    var embeddingPurpose: EmbeddingPurpose { purpose == "query" ? .query : .document }
  }

  /// The contents of one file that `mise run generate:fixtures` wrote.
  private struct ReferenceFixture: Decodable, Sendable {
    let presetModel: String
    let presetCommit: String
    let referenceModel: String
    let referenceCommit: String
    let queryPrefix: String
    let documentPrefix: String
    let maximumTokenCount: Int
    let packages: [String: String]
    let cases: [ReferenceCase]

    static func load(_ key: String) throws -> ReferenceFixture {
      let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/reference-\(key).json")
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      return try decoder.decode(ReferenceFixture.self, from: Data(contentsOf: url))
    }
  }

  /// The agreement that a preset must reach with its reference vectors.
  struct ReferenceTolerance: Sendable {
    /// The lowest allowed cosine similarity with the reference vector.
    let minimumCosine: Float
    /// The largest allowed difference in one element of a unit vector.
    let maximumElementError: Float
  }

  /// A preset and its tolerances.
  struct ReferenceCheck: Sendable, CustomTestStringConvertible {
    let key: String
    let model: MLXEmbeddingModel
    /// The tolerance for ordinary sentences.
    let natural: ReferenceTolerance
    /// The tolerance for inputs of one word repeated up to the token limit.
    let repetitive: ReferenceTolerance

    var testDescription: String { key }

    /// The cases that repeat one word up to the token limit.
    static let repetitiveCases: Set<String> = [
      "below-limit", "at-limit", "above-limit", "far-above-limit", "query-above-limit",
    ]

    func tolerance(for name: String) -> ReferenceTolerance {
      Self.repetitiveCases.contains(name) ? repetitive : natural
    }

    /// The observed agreement of the unquantized presets is better than 1e-6 per element.
    private static let exact = ReferenceTolerance(minimumCosine: 0.99999, maximumElementError: 1e-5)

    static let all = [
      ReferenceCheck(key: "bge-small", model: .bgeSmallEnglish, natural: exact, repetitive: exact),
      ReferenceCheck(key: "minilm", model: .allMiniLM, natural: exact, repetitive: exact),
      ReferenceCheck(key: "nomic", model: .nomicEmbedText, natural: exact, repetitive: exact),
      // The preset is quantized to 4 bits, and its reference is full precision.
      ReferenceCheck(
        key: "qwen3",
        model: .qwen3Embedding(),
        natural: ReferenceTolerance(minimumCosine: 0.955, maximumElementError: 0.04),
        repetitive: ReferenceTolerance(minimumCosine: 0.70, maximumElementError: 0.10)
      ),
    ]
  }

  /// The fixtures are in the repository, so these tests need no network.
  @Suite("MLX reference fixtures")
  struct MLXReferenceFixtureTests {
    @Test("A fixture describes its preset", arguments: ReferenceCheck.all)
    func matchesPreset(check: ReferenceCheck) throws {
      let fixture = try ReferenceFixture.load(check.key)
      #expect(fixture.presetModel == check.model.modelId)
      #expect(fixture.presetCommit == check.model.revision)
      #expect(MLXEmbeddingModel.isCommitHash(fixture.referenceCommit))
      #expect(fixture.queryPrefix == check.model.prefixes.query)
      #expect(fixture.documentPrefix == check.model.prefixes.document)
      #expect(fixture.maximumTokenCount == check.model.maximumTokenCount)
      #expect(fixture.packages["sentence-transformers"] != nil)
      for item in fixture.cases {
        #expect(item.vector.count == check.model.dimension, "\(item.name)")
      }
    }

    @Test(
      "A fixture covers both purposes and the truncation boundaries", arguments: ReferenceCheck.all)
    func coversBoundaries(check: ReferenceCheck) throws {
      let fixture = try ReferenceFixture.load(check.key)
      let limit = fixture.maximumTokenCount
      let counts = Dictionary(
        uniqueKeysWithValues: fixture.cases.map { ($0.name, $0.tokens.count) })
      #expect(counts["below-limit"] == limit - 1)
      #expect(counts["at-limit"] == limit)
      #expect(counts["above-limit"] == limit)
      #expect(counts["far-above-limit"] == limit)
      #expect(counts["query-above-limit"] == limit)
      #expect(Set(fixture.cases.map(\.purpose)) == ["query", "document"])
    }

    @Test("Truncation keeps the last token and reaches the limit")
    func truncation() {
      let ids = Array(0..<10)
      #expect(MLXEmbeddingModel.truncated(ids, limit: 4) == [0, 1, 2, 9])
      #expect(MLXEmbeddingModel.truncated(ids, limit: 10) == ids)
      #expect(MLXEmbeddingModel.truncated(ids, limit: 11) == ids)
      #expect(MLXEmbeddingModel.truncated([], limit: 4) == [])
    }
  }

  /// These tests load real weights.
  @Suite("MLX reference agreement", .serialized, .needsRealModels)
  struct MLXReferenceAgreementTests {
    private func compare(
      _ vector: [Float],
      with item: ReferenceCase,
      tolerance: ReferenceTolerance,
      label: String
    ) -> (cosine: Float, error: Float) {
      let similarity = cosine(vector, item.vector)
      let error = zip(vector, item.vector).map { abs($0 - $1) }.max() ?? 0
      #expect(similarity >= tolerance.minimumCosine, "\(label) \(item.name): \(similarity)")
      #expect(error <= tolerance.maximumElementError, "\(label) \(item.name): \(error)")
      return (similarity, error)
    }

    @Test("Token IDs match the reference tokenizer", arguments: ReferenceCheck.all)
    func tokens(check: ReferenceCheck) async throws {
      let fixture = try ReferenceFixture.load(check.key)
      for item in fixture.cases {
        let ids = try await check.model.tokenIDs(for: [item.text], purpose: item.embeddingPurpose)
        #expect(ids == [item.tokens], "\(check.key) \(item.name)")
      }
    }

    @Test("Qwen3 ends every input with <|endoftext|>, even after truncation")
    func qwen3FinalToken() async throws {
      let endOfText = 151_643
      let imEnd = 151_645
      let model = MLXEmbeddingModel.qwen3Embedding()
      let fixture = try ReferenceFixture.load("qwen3")
      for item in fixture.cases {
        let ids = try await model.tokenIDs(for: [item.text], purpose: item.embeddingPurpose)
        let tokens = try #require(ids.first)
        #expect(tokens.last == endOfText, "\(item.name)")
        #expect(!tokens.contains(imEnd), "\(item.name)")
      }
    }

    @Test("Full-precision Qwen3 weights in this engine match the reference")
    func qwen3FullPrecision() async throws {
      let fixture = try ReferenceFixture.load("qwen3")
      let model = try MLXEmbeddingModel(
        modelId: fixture.referenceModel,
        commit: fixture.referenceCommit,
        dimension: 1024,
        prefixes: .qwen3()
      )
      defer { model.unload() }
      // This shows that the engine is right and that the 4-bit preset's distance
      // from the reference comes from quantization.
      let tolerance = ReferenceTolerance(minimumCosine: 0.9995, maximumElementError: 0.005)
      var worst: (cosine: Float, error: Float) = (1, 0)
      for item in fixture.cases {
        let vector = try await model.embed(item.text, for: item.embeddingPurpose)
        let result = compare(vector, with: item, tolerance: tolerance, label: "full precision")
        worst = (min(worst.cosine, result.cosine), max(worst.error, result.error))
      }
      print("reference qwen3-full: lowest cosine \(worst.cosine), largest error \(worst.error)")
    }

    @Test("Each vector alone matches the reference", arguments: ReferenceCheck.all)
    func singles(check: ReferenceCheck) async throws {
      let fixture = try ReferenceFixture.load(check.key)
      var worst: (cosine: Float, error: Float) = (1, 0)
      for item in fixture.cases {
        let vector = try await check.model.embed(item.text, for: item.embeddingPurpose)
        let result = compare(
          vector, with: item, tolerance: check.tolerance(for: item.name),
          label: "\(check.key) alone")
        worst = (min(worst.cosine, result.cosine), max(worst.error, result.error))
      }
      print(
        "reference \(check.key) alone: lowest cosine \(worst.cosine), largest error \(worst.error)")
    }

    @Test("Vectors in unequal-length batches match the reference", arguments: ReferenceCheck.all)
    func batches(check: ReferenceCheck) async throws {
      let fixture = try ReferenceFixture.load(check.key)
      var worst: (cosine: Float, error: Float) = (1, 0)
      for purpose in EmbeddingPurpose.allCases {
        let items = fixture.cases.filter { $0.embeddingPurpose == purpose }
        let vectors = try await check.model.embed(items.map(\.text), for: purpose)
        for (item, vector) in zip(items, vectors) {
          let result = compare(
            vector, with: item, tolerance: check.tolerance(for: item.name),
            label: "\(check.key) batch")
          worst = (min(worst.cosine, result.cosine), max(worst.error, result.error))
        }
      }
      print(
        "reference \(check.key) batch: lowest cosine \(worst.cosine), largest error \(worst.error)")
    }
  }
#endif
