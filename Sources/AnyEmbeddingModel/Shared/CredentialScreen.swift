import Foundation

/// Decides whether text from a server might disclose a credential.
///
/// A server can echo the credential that it rejected,
/// and it can encode that credential in many equivalent ways.
/// So the screen doesn't look for a list of encoded forms.
/// It decodes the text instead:
/// JSON escapes, including `\uXXXX`, and percent-encoding in either case,
/// repeatedly, for text that is encoded more than once.
/// After each pass, it looks for the credential in the bytes,
/// ignoring ASCII case and allowing composed and decomposed Unicode forms.
///
/// The text isn't clear if the credential appears,
/// if the text isn't valid UTF-8,
/// or if decoding doesn't finish within ``maximumPasses``.
///
/// The screen can't detect:
/// - part of a credential, such as a masked or truncated copy;
/// - a credential in another encoding, such as Base64, hexadecimal,
///   HTML entities, or `%uXXXX`;
/// - a credential with other characters inserted between its characters;
/// - a credential that differs by a case change outside ASCII.
struct CredentialScreen: Sendable {
  /// The number of decoding passes before the text counts as unclear.
  static let maximumPasses = 8

  /// The forms of the credential to look for, in folded ASCII case.
  private let needles: [[UInt8]]

  init(credential: String) {
    let forms = [
      credential,
      credential.precomposedStringWithCanonicalMapping,
      credential.decomposedStringWithCanonicalMapping,
    ]
    needles = Array(Set(forms.map { Self.folded(Array($0.utf8)) })).filter { !$0.isEmpty }
  }

  /// Returns whether `bytes` can be shown without disclosing the credential,
  /// as far as the screen can detect.
  func isClear(_ bytes: [UInt8]) -> Bool {
    guard String(bytes: bytes, encoding: .utf8) != nil else { return false }
    var current = bytes
    for _ in 0..<Self.maximumPasses {
      if discloses(current) { return false }
      let next = Self.decodingJSONEscapes(Self.decodingPercentEscapes(current))
      if next == current { return true }
      current = next
    }
    return false
  }

  private func discloses(_ bytes: [UInt8]) -> Bool {
    let haystack = Self.folded(bytes)
    return needles.contains { needle in
      guard haystack.count >= needle.count else { return false }
      return (0...(haystack.count - needle.count)).contains { start in
        haystack[start..<(start + needle.count)].elementsEqual(needle)
      }
    }
  }

  private static func folded(_ bytes: [UInt8]) -> [UInt8] {
    bytes.map { (0x41...0x5A).contains($0) ? $0 + 0x20 : $0 }
  }

  private static func hexValue(_ byte: UInt8) -> UInt8? {
    switch byte {
    case 0x30...0x39: byte - 0x30
    case 0x41...0x46: byte - 0x41 + 10
    case 0x61...0x66: byte - 0x61 + 10
    default: nil
    }
  }

  /// Replaces each `%XX`, in either case, and leaves other bytes as they are.
  private static func decodingPercentEscapes(_ bytes: [UInt8]) -> [UInt8] {
    var decoded: [UInt8] = []
    decoded.reserveCapacity(bytes.count)
    var index = 0
    while index < bytes.count {
      if bytes[index] == 0x25, index + 2 < bytes.count,
        let high = hexValue(bytes[index + 1]), let low = hexValue(bytes[index + 2])
      {
        decoded.append(high << 4 | low)
        index += 3
      } else {
        decoded.append(bytes[index])
        index += 1
      }
    }
    return decoded
  }

  /// Returns the code unit of the `\uXXXX` escape at `index`, if there is one.
  private static func unicodeEscape(in bytes: [UInt8], at index: Int) -> UInt16? {
    guard index + 5 < bytes.count, bytes[index] == 0x5C, bytes[index + 1] == 0x75 else {
      return nil
    }
    var value: UInt16 = 0
    for offset in 2...5 {
      guard let digit = hexValue(bytes[index + offset]) else { return nil }
      value = value << 4 | UInt16(digit)
    }
    return value
  }

  /// Replaces JSON's escape sequences, and leaves other bytes as they are.
  private static func decodingJSONEscapes(_ bytes: [UInt8]) -> [UInt8] {
    let simple: [UInt8: UInt8] = [
      0x22: 0x22, 0x5C: 0x5C, 0x2F: 0x2F, 0x62: 0x08, 0x66: 0x0C, 0x6E: 0x0A, 0x72: 0x0D,
      0x74: 0x09,
    ]
    var decoded: [UInt8] = []
    decoded.reserveCapacity(bytes.count)
    var index = 0
    while index < bytes.count {
      if let unit = unicodeEscape(in: bytes, at: index) {
        var units = [unit]
        index += 6
        if UTF16.isLeadSurrogate(unit), let trail = unicodeEscape(in: bytes, at: index),
          UTF16.isTrailSurrogate(trail)
        {
          units.append(trail)
          index += 6
        }
        decoded += Array(String(decoding: units, as: UTF16.self).utf8)
      } else if bytes[index] == 0x5C, index + 1 < bytes.count, let value = simple[bytes[index + 1]]
      {
        decoded.append(value)
        index += 2
      } else {
        decoded.append(bytes[index])
        index += 1
      }
    }
    return decoded
  }
}
