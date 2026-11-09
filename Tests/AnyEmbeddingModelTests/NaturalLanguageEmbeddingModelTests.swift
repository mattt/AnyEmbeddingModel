#if canImport(NaturalLanguage)
  import NaturalLanguage
  import Testing

  import AnyEmbeddingModel

  // A machine can lack the system's English sentence embedding.
  @Suite(
    "NaturalLanguageEmbeddingModel",
    .enabled(if: NaturalLanguageEmbeddingModel(language: .english) != nil)
  )
  struct NaturalLanguageEmbeddingModelTests {
    @Test("Returns unit vectors that rank a related text above an unrelated one")
    func ranking() async throws {
      let model = try #require(NaturalLanguageEmbeddingModel(language: .english))
      #expect(model.isAvailable)
      let documents = try await model.embed(
        ["My sister lives in Lisbon.", "The printer is out of toner."],
        for: .document
      )
      let query = try await model.embed("Where does my sister live?", for: .query)

      for vector in documents + [query] {
        #expect(vector.count == model.dimension)
        let length = vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
        #expect(abs(length - 1) < 1e-4)
      }
      let similarities = documents.map { zip($0, query).reduce(0) { $0 + $1.0 * $1.1 } }
      #expect(similarities[0] > similarities[1])
    }

    @Test("Prefixes change the vector and the identifier")
    func prefixes() async throws {
      let plain = try #require(NaturalLanguageEmbeddingModel(language: .english))
      let prefixed = try #require(
        NaturalLanguageEmbeddingModel(
          language: .english,
          prefixes: EmbeddingPrefixes(query: "Question: ")
        ))
      #expect(plain.identifier != prefixed.identifier)
      #expect(plain.identifier.hasPrefix("naturallanguage/sentence-en/"))
      let a = try await plain.embed("Where are my keys?", for: .query)
      let b = try await prefixed.embed("Where are my keys?", for: .query)
      let c = try await prefixed.embed("Where are my keys?", for: .document)
      #expect(a != b)
      #expect(a == c)
    }

    /// The revision and dimension come from the system,
    /// and the fingerprint comes from `scripts/generate-identifier-goldens.py`.
    @Test("The identifier matches the independent implementation")
    func identifierGolden() throws {
      let model = try #require(NaturalLanguageEmbeddingModel(language: .english))
      #expect(
        model.identifier
          == "naturallanguage/sentence-en/\(model.revision)/\(model.dimension)/cf21924e7b0ff7c7")
    }

    @Test("Rejects empty text")
    func emptyText() async throws {
      let model = try #require(NaturalLanguageEmbeddingModel(language: .english))
      await #expect(throws: EmbeddingError.emptyText(index: 0)) {
        try await model.embed([""], for: .document)
      }
      #expect(try await model.embed([], for: .document).isEmpty)
    }
  }
#endif
