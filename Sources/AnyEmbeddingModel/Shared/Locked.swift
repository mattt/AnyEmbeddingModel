import Foundation

/// A value that one thread at a time can read or change.
///
/// `Synchronization.Mutex` needs macOS 15 and iOS 18,
/// and this package supports macOS 14 and iOS 17.
final class Locked<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Value

  init(_ value: Value) {
    self.value = value
  }

  func withLock<Result>(_ body: (inout Value) throws -> Result) rethrows -> Result {
    lock.lock()
    defer { lock.unlock() }
    return try body(&value)
  }
}
