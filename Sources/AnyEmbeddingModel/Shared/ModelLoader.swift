/// Identifies the files that a ``ModelLoader`` loads.
///
/// The fields are separate, so that no text in a path, a name, or a version
/// can make two different sources look alike.
struct ModelLoaderKey: Hashable, Sendable {
    /// Where the files come from.
    enum Source: Hashable, Sendable {
        /// A repository on a model hub.
        case hub

        /// A directory on disk.
        case directory
    }

    let source: Source

    /// The repository of a hub model, or the path of a directory.
    let location: String

    /// The commit of a hub model, or the version of a directory.
    let version: String
}

/// Loads a value once and shares it between callers.
///
/// Each load has a generation.
/// A load's own task publishes its result, exactly once,
/// and only if its generation is still the current one.
/// So a load that ``unload()`` abandoned,
/// or that a newer load replaced,
/// can't change the loader's state when it finally ends,
/// even if it ignores cancellation.
final class ModelLoader<Container: Sendable>: Sendable {
    enum State: Equatable {
        case notLoaded
        case loaded
        case failed(String)
    }

    private struct Storage {
        /// Identifies the current load; ``unload()`` also advances it.
        var generation: UInt64 = 0
        var container: Container?
        var task: Task<Container, any Error>?
        var failure: String?
        /// The number of callers that have joined the current load.
        var waiters = 0
    }

    private enum Step {
        case ready(Container)
        case wait(Task<Container, any Error>)
    }

    private let load: @Sendable () async throws -> Container
    private let storage = Locked(Storage())

    init(load: @escaping @Sendable () async throws -> Container) {
        self.load = load
    }

    var state: State {
        storage.withLock { storage in
            if storage.container != nil { return .loaded }
            if let failure = storage.failure { return .failed(failure) }
            return .notLoaded
        }
    }

    /// The number of callers that have joined the load that is running now.
    var waiterCount: Int {
        storage.withLock { $0.waiters }
    }

    /// Returns the loaded value, and starts one load if none is running.
    ///
    /// Callers that arrive during a load wait for that load.
    /// A caller receives the result of the load that it waited for,
    /// even if ``unload()`` abandoned that load in the meantime.
    func container() async throws -> Container {
        let step = storage.withLock { storage -> Step in
            if let container = storage.container { return .ready(container) }
            if let task = storage.task {
                storage.waiters += 1
                return .wait(task)
            }
            storage.generation &+= 1
            storage.failure = nil
            storage.waiters = 1
            let generation = storage.generation
            let task = Task { [self] in
                do {
                    let container = try await load()
                    publish(generation) { $0.container = container }
                    return container
                } catch {
                    let failure = String(describing: error)
                    publish(generation) { $0.failure = failure }
                    throw error
                }
            }
            storage.task = task
            return .wait(task)
        }
        switch step {
        case .ready(let container): return container
        case .wait(let task): return try await task.value
        }
    }

    /// Abandons any running load and releases the value.
    ///
    /// The next call to ``container()`` starts a new load.
    func unload() {
        storage.withLock { storage in
            storage.generation &+= 1
            storage.container = nil
            storage.failure = nil
            storage.waiters = 0
            storage.task?.cancel()
            storage.task = nil
        }
    }

    /// Applies the result of a load, if that load is still the current one.
    private func publish(_ generation: UInt64, _ result: (inout Storage) -> Void) {
        storage.withLock { storage in
            guard storage.generation == generation else { return }
            storage.task = nil
            storage.waiters = 0
            result(&storage)
        }
    }
}
