import Foundation

/// Protects shared mutable state behind an `NSLock`.
///
/// `Synchronization.Mutex` needs macOS 15 and iOS 18,
/// and this package supports macOS 14 and iOS 17.
/// Like AnyLanguageModel's `Locked`,
/// it is `Sendable` only when its state is,
/// so it can't hide a value that isn't safe to share.
final class Locked<State> {
    private let lock = NSLock()
    private var state: State

    /// Creates a locked container with the given initial state.
    init(_ state: State) {
        self.state = state
    }

    /// Executes `body` while holding the lock.
    ///
    /// Keep the body small and synchronous.
    func withLock<T>(_ body: (inout State) throws -> T) rethrows -> T {
        try lock.withLock { try body(&self.state) }
    }
}

extension Locked: @unchecked Sendable where State: Sendable {}
