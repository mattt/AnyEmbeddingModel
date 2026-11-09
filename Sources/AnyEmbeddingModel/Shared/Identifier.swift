/// Builds the ``EmbeddingModel/identifier`` of a model.
///
/// An identifier has readable fields, separated by slashes,
/// for the engine, the model, its revision, and the dimension.
/// The last field is a fingerprint of the other settings that change vectors,
/// such as the prefixes,
/// so that the identifier stays short but still changes when they change.
///
/// Two different sets of values never have the same encoding.
/// Each field is the percent-encoding of its UTF-8 bytes,
/// and only ASCII letters, digits, `-`, `.`, `_`, and `~` stay as they are.
/// So an identifier is pure ASCII,
/// the slashes between fields are the only slashes,
/// and a combining mark can't attach to a slash and hide it.
/// The fingerprint hashes each setting with its length.
///
/// Identity is by bytes.
/// The package doesn't normalize Unicode,
/// so composed and decomposed forms of the same text are different identifiers,
/// even though Swift compares those strings as equal.
/// The two forms reach a tokenizer as different bytes,
/// and they can produce different vectors.
///
/// For example:
/// `mlx/BAAI%2Fbge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/384/17ad23889b01fd7d`.
enum Identifier {
  /// The field for a model with no revision.
  ///
  /// A revision of exactly this value is percent-encoded, so that the two can't collide.
  private static let noRevision = "-"

  static func make(
    engine: String,
    model: String,
    revision: String?,
    dimension: Int,
    settings: [String]
  ) -> String {
    let revisionField = revision.map(encodeRevision) ?? noRevision
    let fields = [
      encode(engine), encode(model), revisionField, String(dimension), fingerprint(settings),
    ]
    return fields.joined(separator: "/")
  }

  /// Returns the percent-encoding of the UTF-8 bytes of `field`.
  ///
  /// The hexadecimal digits are uppercase,
  /// so a field has exactly one encoding.
  static func encode(_ field: String) -> String {
    var encoded = ""
    for byte in field.utf8 {
      if isUnreserved(byte) {
        encoded.unicodeScalars.append(Unicode.Scalar(byte))
      } else {
        let digits = String(byte, radix: 16, uppercase: true)
        encoded += "%" + (digits.count == 1 ? "0" : "") + digits
      }
    }
    return encoded
  }

  /// Returns whether `byte` is an ASCII letter or digit, or one of `-`, `.`, `_`, and `~`.
  private static func isUnreserved(_ byte: UInt8) -> Bool {
    switch byte {
    case 0x30...0x39, 0x41...0x5A, 0x61...0x7A, 0x2D, 0x2E, 0x5F, 0x7E: true
    default: false
    }
  }

  private static func encodeRevision(_ revision: String) -> String {
    revision == noRevision ? "%2D" : encode(revision)
  }

  /// Returns the 64-bit FNV-1a hash of the settings, in hexadecimal.
  ///
  /// The hash is stable across processes and platforms,
  /// unlike `Hasher`, which uses a random seed.
  /// The input has the number of settings,
  /// then each setting's UTF-8 length followed by its bytes,
  /// so `["ab", "c"]`, `["a", "bc"]`, and `["a\0", "b"]` are all different inputs.
  static func fingerprint(_ settings: [String]) -> String {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    func add(_ byte: UInt8) {
      hash ^= UInt64(byte)
      hash &*= 0x0000_0100_0000_01b3
    }
    func add(length: Int) {
      withUnsafeBytes(of: UInt64(length).littleEndian) { $0.forEach(add) }
    }
    add(length: settings.count)
    for setting in settings {
      add(length: setting.utf8.count)
      setting.utf8.forEach(add)
    }
    let digits = String(hash, radix: 16)
    return String(repeating: "0", count: 16 - digits.count) + digits
  }
}
