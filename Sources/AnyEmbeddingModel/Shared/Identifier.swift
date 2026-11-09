/// Builds the ``EmbeddingModel/identifier`` of a model.
///
/// An identifier has readable fields, separated by slashes,
/// for the engine, the model, its revision, and the dimension.
/// The last field is a fingerprint of the other settings that change vectors,
/// such as the prefixes,
/// so that the identifier stays short but still changes when they change.
///
/// Two different sets of values never have the same encoding:
/// a slash inside a field is percent-encoded,
/// so the slashes between fields are the only slashes,
/// and the fingerprint hashes each setting with its length.
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

  /// Returns `field` with the characters that would make its boundaries unclear
  /// percent-encoded.
  ///
  /// Only `%` and `/` change, so that most fields stay as they are.
  static func encode(_ field: String) -> String {
    field.replacing("%", with: "%25").replacing("/", with: "%2F")
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
