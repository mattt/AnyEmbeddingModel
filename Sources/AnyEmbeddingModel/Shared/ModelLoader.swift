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

  /// Returns the loaded value, and starts one load if none is running.
  ///
  /// Callers that arrive during a load wait for that load.
  /// A caller receives the result of the load that it waited for,
  /// even if ``unload()`` abandoned that load in the meantime.
  func container() async throws -> Container {
    let step = storage.withLock { storage -> Step in
      if let container = storage.container { return .ready(container) }
      if let task = storage.task { return .wait(task) }
      storage.generation &+= 1
      storage.failure = nil
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
      storage.task?.cancel()
      storage.task = nil
    }
  }

  /// Applies the result of a load, if that load is still the current one.
  private func publish(_ generation: UInt64, _ result: (inout Storage) -> Void) {
    storage.withLock { storage in
      guard storage.generation == generation else { return }
      storage.task = nil
      result(&storage)
    }
  }
}
