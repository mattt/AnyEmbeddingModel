import Testing

@testable import AnyEmbeddingModel

@Suite("CredentialScreen")
struct CredentialScreenTests {
    private let screen = CredentialScreen(credential: "sk-secret/one+two")

    private func isClear(_ text: String) -> Bool {
        screen.isClear(Array(text.utf8))
    }

    @Test(
        "Text without the credential is clear",
        arguments: [
            "", "Rate limit exceeded.", "100% of capacity", "%zz", #"C:\users\name"#, "sk-secret",
        ]
    )
    func clearText(text: String) {
        #expect(isClear(text))
    }

    @Test(
        "The credential is found in equivalent encodings",
        arguments: [
            "sk-secret/one+two",
            "SK-SECRET/ONE+TWO",
            #"sk-secret\/one+two"#,
            #"\u0073k-secret/one+two"#,
            #"\u0073K-SECRET\u002fone\u002Btwo"#,
            "sk%2Dsecret%2Fone%2Btwo",
            "sk-secret%2fone%2btwo",
            "sk-secret%252fone%252btwo",
            #"sk-secret\u00252fone+two"#,
        ]
    )
    func detected(text: String) {
        #expect(!isClear(text))
    }

    @Test("Text that isn't valid UTF-8 isn't clear")
    func invalidUTF8() {
        #expect(!screen.isClear([0x61, 0xFF, 0x62]))
    }

    @Test("Text that needs too many decoding passes isn't clear")
    func tooManyPasses() {
        var text = "%41"
        for _ in 0 ..< CredentialScreen.maximumPasses {
            text = text.replacing("%", with: "%25")
        }
        #expect(!isClear(text))
    }

    @Test("Text encoded fewer times than the limit is decoded")
    func nestedWithinLimit() {
        var text = "sk-secret%2Fone+two"
        for _ in 0 ..< 3 {
            text = text.replacing("%", with: "%25")
        }
        #expect(!isClear(text))
    }

    @Test("An empty credential matches nothing")
    func emptyCredential() {
        #expect(CredentialScreen(credential: "").isClear(Array("anything".utf8)))
    }

    @Test("A credential in composed form is found in decomposed text, and the reverse")
    func unicodeForms() {
        let composed = CredentialScreen(credential: "cl\u{E9}")
        #expect(!composed.isClear(Array("the cle\u{301} here".utf8)))
        let decomposed = CredentialScreen(credential: "cle\u{301}")
        #expect(!decomposed.isClear(Array("the cl\u{E9} here".utf8)))
    }

    @Test("A form that one decoding stage produces is checked before the next stage changes it")
    func checksEachStage() {
        // The credential has a literal backslash and `n`.
        let screen = CredentialScreen(credential: #"sk-\n"#)
        // Percent decoding gives the credential, and JSON decoding would turn it into a newline.
        #expect(!screen.isClear(Array("sk-%5Cn".utf8)))
        #expect(!screen.isClear(Array("sk-%5cn".utf8)))
        // JSON decoding gives the credential, and percent decoding in the next pass would change it.
        let percent = CredentialScreen(credential: "sk-%41")
        #expect(!percent.isClear(Array(#"sk-\u002541"#.utf8)))
    }

    @Test("A stage that produces text that isn't valid UTF-8 isn't clear")
    func invalidUTF8AfterDecoding() {
        #expect(!screen.isClear(Array("bad %FF byte".utf8)))
        #expect(!screen.isClear(Array("bad %C3 byte".utf8)))
    }

    @Test("A credential echoed with mixed composed and decomposed characters is found")
    func mixedUnicodeForms() {
        let screen = CredentialScreen(credential: "k\u{E9}y-\u{E9}t\u{E9}")
        #expect(!screen.isClear(Array("echo k\u{E9}y-\u{E9}te\u{301} end".utf8)))
        #expect(!screen.isClear(Array("echo ke\u{301}y-\u{E9}t\u{E9} end".utf8)))
        // The decomposed character can also arrive percent-encoded.
        #expect(!screen.isClear(Array("echo k%C3%A9y-%C3%A9te%CC%81 end".utf8)))
        #expect(screen.isClear(Array("echo key-ete end".utf8)))
    }
}
