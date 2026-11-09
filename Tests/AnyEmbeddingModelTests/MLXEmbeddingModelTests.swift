#if MLX
  import Foundation
  import Testing

  @testable import AnyEmbeddingModel

  /// These tests download `BAAI/bge-small-en-v1.5` from the Hugging Face Hub,
  /// about 130 MB, on the first run, at the commit that the preset pins.
  @Suite("MLXEmbeddingModel", .serialized, .needsRealModels)
  struct MLXEmbeddingModelTests {
    @Test("Loads, then returns unit vectors that rank a related text first")
    func ranking() async throws {
      let model = MLXEmbeddingModel.bgeSmallEnglish
      #expect(model.availability == .unavailable(.notLoaded))
      try await model.prewarm()
      #expect(model.isAvailable)

      let documents = try await model.embed(
        ["My sister lives in Lisbon.", "The printer is out of toner.", "Buy oat milk."],
        for: .document
      )
      let query = try await model.embed("Which city is my sibling in?", for: .query)
      for vector in documents + [query] {
        #expect(vector.count == 384)
        let length = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        #expect(abs(length - 1) < 1e-3)
      }
      let similarities = documents.map { zip($0, query).reduce(0) { $0 + $1.0 * $1.1 } }
      // Strictly greater, so that a tie among the documents fails.
      #expect(similarities[0] > similarities[1])
      #expect(similarities[0] > similarities[2])
    }

    @Test("A batch gives the same vectors as single texts, despite padding")
    func batchMatchesSingle() async throws {
      let model = MLXEmbeddingModel.bgeSmallEnglish
      let texts = ["Short.", "A much longer sentence that needs more tokens than the first one."]
      let batch = try await model.embed(texts, for: .document)
      for (text, vector) in zip(texts, batch) {
        let single = try await model.embed(text, for: .document)
        let similarity = zip(single, vector).reduce(0) { $0 + $1.0 * $1.1 }
        #expect(similarity > 0.9999)
      }
    }

    @Test("Prewarming reports a wrong dimension")
    func wrongDimension() async throws {
      let model = try MLXEmbeddingModel(
        modelId: "BAAI/bge-small-en-v1.5",
        commit: MLXEmbeddingModel.bgeSmallEnglish.revision,
        dimension: 383
      )
      await #expect(throws: EmbeddingError.dimensionMismatch(expected: 383, actual: 384)) {
        try await model.prewarm()
      }
    }

    @Test("Truncates a long text instead of failing")
    func truncation() async throws {
      let model = try MLXEmbeddingModel(
        modelId: "BAAI/bge-small-en-v1.5",
        commit: MLXEmbeddingModel.bgeSmallEnglish.revision,
        dimension: 384,
        prefixes: .bge,
        maximumTokenCount: 16
      )
      let long = String(repeating: "tea and biscuits ", count: 200)
      let vector = try await model.embed(long, for: .document)
      #expect(vector.count == 384)
      #expect(model.identifier != MLXEmbeddingModel.bgeSmallEnglish.identifier)

      // BERT's first token is [CLS], and its last is [SEP], which truncation keeps.
      let tokens = try #require(try await model.tokenIDs(for: [long], purpose: .document).first)
      let untruncated = try #require(
        try await MLXEmbeddingModel.bgeSmallEnglish.tokenIDs(for: [long], purpose: .document).first
      )
      #expect(tokens.count == 16)
      #expect(tokens.first == 101)
      #expect(tokens.last == 102)
      #expect(tokens.dropLast() == untruncated.prefix(15))
      #expect(untruncated.count == 512)

      let full = try await MLXEmbeddingModel.bgeSmallEnglish.embed(long, for: .document)
      #expect(cosine(vector, full) < 0.9999)
    }
  }
#endif
