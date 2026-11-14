import Testing

@testable import AnyEmbeddingModel

@Suite("Core")
struct CoreTests {
    @Test("Prefixes go before the text exactly as written")
    func prefixes() {
        #expect(EmbeddingPrefixes.nomic.apply(to: "tea", for: .query) == "search_query: tea")
        #expect(EmbeddingPrefixes.nomic.apply(to: "tea", for: .document) == "search_document: tea")
        #expect(EmbeddingPrefixes.bge.apply(to: "tea", for: .document) == "tea")
        #expect(EmbeddingPrefixes.none.apply(to: "tea", for: .query) == "tea")
    }

    @Test("An identifier changes when a setting changes")
    func identifierChanges() {
        let a = Identifier.make(
            engine: "e",
            model: "m",
            revision: "1",
            dimension: 3,
            settings: ["ab", "c"]
        )
        let b = Identifier.make(
            engine: "e",
            model: "m",
            revision: "1",
            dimension: 3,
            settings: ["a", "bc"]
        )
        let c = Identifier.make(
            engine: "e",
            model: "m",
            revision: "1",
            dimension: 3,
            settings: ["ab", "c"]
        )
        #expect(a != b)
        #expect(a == c)
        #expect(a.hasPrefix("e/m/1/3/"))
    }

    @Test("The fingerprint is stable across runs")
    func fingerprintIsStable() {
        // Golden values: a change here changes every saved identifier.
        #expect(Identifier.fingerprint([]) == "a8c7f832281a39c5")
        #expect(Identifier.fingerprint(["query: ", "passage: "]) == "080fecb3afe4ad77")
    }

    @Test("Settings with the same bytes in different fields have different fingerprints")
    func fingerprintFraming() {
        #expect(Identifier.fingerprint(["a\0", "b"]) != Identifier.fingerprint(["a", "\0b"]))
        #expect(Identifier.fingerprint(["ab", "c"]) != Identifier.fingerprint(["a", "bc"]))
        #expect(Identifier.fingerprint(["", "a"]) != Identifier.fingerprint(["a", ""]))
        #expect(Identifier.fingerprint(["a"]) != Identifier.fingerprint(["a", ""]))
    }

    @Test("A slash inside a field can't collide with the slash between fields")
    func fieldFraming() {
        let a = Identifier.make(
            engine: "e",
            model: "org/m",
            revision: "v/1",
            dimension: 3,
            settings: []
        )
        let b = Identifier.make(
            engine: "e",
            model: "org/m/v",
            revision: "1",
            dimension: 3,
            settings: []
        )
        #expect(a != b)
        #expect(a.hasPrefix("e/org%2Fm/v%2F1/3/"))
        #expect(a.split(separator: "/").count == 5)
    }

    @Test("A percent sign is encoded, so an encoded slash is distinct from a literal one")
    func percentFraming() {
        let literal = Identifier.make(
            engine: "e",
            model: "a%2Fb",
            revision: "1",
            dimension: 3,
            settings: []
        )
        let slash = Identifier.make(
            engine: "e",
            model: "a/b",
            revision: "1",
            dimension: 3,
            settings: []
        )
        #expect(literal != slash)
    }

    @Test("A combining mark can't hide a field boundary")
    func combiningMarkFraming() {
        let a = Identifier.make(
            engine: "e",
            model: "a/\u{301}b",
            revision: "\u{301}c",
            dimension: 3,
            settings: []
        )
        let b = Identifier.make(
            engine: "e",
            model: "a",
            revision: "\u{301}b/\u{301}c",
            dimension: 3,
            settings: []
        )
        #expect(Array(a.utf8) != Array(b.utf8))
        #expect(a.utf8.allSatisfy { $0 < 128 })
        #expect(a.split(separator: "/").count == 5)
        #expect(b.split(separator: "/").count == 5)
    }

    @Test("Composed and decomposed text are different identifiers")
    func unicodeNormalization() {
        let composed = Identifier.make(
            engine: "e",
            model: "mod\u{E8}le",
            revision: nil,
            dimension: 3,
            settings: []
        )
        let decomposed = Identifier.make(
            engine: "e",
            model: "mode\u{300}le",
            revision: nil,
            dimension: 3,
            settings: []
        )
        #expect(composed != decomposed)
        #expect(composed.hasPrefix("e/mod%C3%A8le/"))
        #expect(decomposed.hasPrefix("e/mode%CC%80le/"))
        #expect(Identifier.fingerprint(["\u{E8}"]) != Identifier.fingerprint(["e\u{300}"]))
    }

    @Test("No revision is distinct from a revision that looks like the placeholder")
    func missingRevision() {
        let none = Identifier.make(engine: "e", model: "m", revision: nil, dimension: 3, settings: [])
        let dash = Identifier.make(engine: "e", model: "m", revision: "-", dimension: 3, settings: [])
        let empty = Identifier.make(engine: "e", model: "m", revision: "", dimension: 3, settings: [])
        #expect(Set([none, dash, empty]).count == 3)
        #expect(none.hasPrefix("e/m/-/3/"))
    }

    @Test("Normalization returns unit vectors and rejects bad ones")
    func normalization() throws {
        let vector = try Vectors.normalized([3, 4], dimension: 2, index: 0)
        #expect(vector == [0.6, 0.8])
        #expect(throws: EmbeddingError.invalidVector(index: 0)) {
            try Vectors.normalized([0, 0], dimension: 2, index: 0)
        }
        #expect(throws: EmbeddingError.invalidVector(index: 1)) {
            try Vectors.normalized([.nan, 1], dimension: 2, index: 1)
        }
        #expect(throws: EmbeddingError.dimensionMismatch(expected: 3, actual: 2)) {
            try Vectors.normalized([1, 2], dimension: 3, index: 0)
        }
    }

    @Test("An invalid vector's index is relative to the whole input")
    func invalidVectorIndex() {
        #expect(throws: EmbeddingError.invalidVector(index: 33)) {
            try Vectors.normalized([[1, 0], [0, 0]], count: 2, dimension: 2, firstIndex: 32)
        }
    }

    @Test("Validation rejects whitespace-only text")
    func validation() throws {
        try Vectors.validate(["a", "b"])
        #expect(throws: EmbeddingError.emptyText(index: 1)) {
            try Vectors.validate(["a", " \n\t"])
        }
    }

    @Test("Batches keep order and size")
    func batches() {
        let batches = Vectors.batches(Array(0 ..< 5), size: 2).map(Array.init)
        #expect(batches == [[0, 1], [2, 3], [4]])
    }
}
