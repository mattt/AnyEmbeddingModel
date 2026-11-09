#if canImport(Network)
  import Testing

  @testable import AnyEmbeddingModel

  /// A load that ends only when a test tells it to, and ignores cancellation.
  private final class StubLoad: Sendable {
    private struct Failure: Error, CustomStringConvertible {
      var description: String { "stub failure" }
    }

    private let continuations = TestLock<[CheckedContinuation<Int, any Error>?]>([])

    /// Each load's value is its position, starting at 0.
    func load() async throws -> Int {
      try await withCheckedThrowingContinuation { continuation in
        continuations.withLock { $0.append(continuation) }
      }
    }

    var started: Int { continuations.withLock { $0.count } }

    func waitForStart(_ count: Int) async {
      while started < count { await Task.yield() }
    }

    func succeed(_ index: Int) {
      take(index).resume(returning: index)
    }

    func fail(_ index: Int) {
      take(index).resume(throwing: Failure())
    }

    private func take(_ index: Int) -> CheckedContinuation<Int, any Error> {
      continuations.withLock { continuations in
        defer { continuations[index] = nil }
        return continuations[index]!
      }
    }
  }

  @Suite("ModelLoader")
  struct ModelLoaderTests {
    private let stub: StubLoad
    private let loader: ModelLoader<Int>

    init() {
      let stub = StubLoad()
      self.stub = stub
      loader = ModelLoader { try await stub.load() }
    }

    /// Waits until the state changes, because a load publishes from its own task.
    private func settle(to expected: ModelLoader<Int>.State) async {
      while loader.state != expected { await Task.yield() }
    }

    @Test("Several waiters share one load")
    func waitersShareALoad() async throws {
      let waiters = (0..<5).map { _ in Task { try await loader.container() } }
      await stub.waitForStart(1)
      stub.succeed(0)
      for waiter in waiters {
        #expect(try await waiter.value == 0)
      }
      #expect(stub.started == 1)
      #expect(loader.state == .loaded)
    }

    @Test("A loaded value is reused")
    func reuse() async throws {
      let first = Task { try await loader.container() }
      await stub.waitForStart(1)
      stub.succeed(0)
      _ = try await first.value
      #expect(try await loader.container() == 0)
      #expect(stub.started == 1)
    }

    @Test("Unloading during a load leaves the loader unloaded when the load ends")
    func unloadDuringLoad() async throws {
      let waiter = Task { try await loader.container() }
      await stub.waitForStart(1)
      loader.unload()
      #expect(loader.state == .notLoaded)

      stub.succeed(0)
      // The waiter still gets the result of the load that it waited for.
      #expect(try await waiter.value == 0)
      #expect(loader.state == .notLoaded)
    }

    @Test("An abandoned failure doesn't report a failure or erase a newer load")
    func abandonedFailure() async throws {
      let abandoned = Task { try await loader.container() }
      await stub.waitForStart(1)
      loader.unload()

      let current = Task { try await loader.container() }
      await stub.waitForStart(2)

      stub.fail(0)
      await #expect(throws: (any Error).self) { try await abandoned.value }
      #expect(loader.state == .notLoaded)

      // A new caller joins the current load and doesn't start a third.
      let joined = Task { try await loader.container() }
      await Task.yield()
      stub.succeed(1)
      #expect(try await current.value == 1)
      #expect(try await joined.value == 1)
      #expect(stub.started == 2)
      #expect(loader.state == .loaded)
    }

    @Test("An abandoned success doesn't replace a newer value")
    func abandonedSuccess() async throws {
      let abandoned = Task { try await loader.container() }
      await stub.waitForStart(1)
      loader.unload()

      let current = Task { try await loader.container() }
      await stub.waitForStart(2)
      stub.succeed(1)
      #expect(try await current.value == 1)

      stub.succeed(0)
      #expect(try await abandoned.value == 0)
      #expect(try await loader.container() == 1)
      #expect(stub.started == 2)
    }

    @Test("A failure is reported, and the next call tries again")
    func retryAfterFailure() async throws {
      let first = Task { try await loader.container() }
      await stub.waitForStart(1)
      stub.fail(0)
      await #expect(throws: (any Error).self) { try await first.value }
      await settle(to: .failed("stub failure"))

      let second = Task { try await loader.container() }
      await stub.waitForStart(2)
      #expect(loader.state == .notLoaded)
      stub.succeed(1)
      #expect(try await second.value == 1)
      #expect(loader.state == .loaded)
    }

    @Test("All waiters on a failed load fail, and the state reports one failure")
    func waitersOnAFailure() async throws {
      let waiters = (0..<3).map { _ in Task { try await loader.container() } }
      await stub.waitForStart(1)
      stub.fail(0)
      for waiter in waiters {
        await #expect(throws: (any Error).self) { try await waiter.value }
      }
      #expect(stub.started == 1)
      #expect(loader.state == .failed("stub failure"))
    }

    @Test("Unloading a loaded value makes the next call load again")
    func unloadThenReload() async throws {
      let first = Task { try await loader.container() }
      await stub.waitForStart(1)
      stub.succeed(0)
      _ = try await first.value

      loader.unload()
      #expect(loader.state == .notLoaded)

      let second = Task { try await loader.container() }
      await stub.waitForStart(2)
      stub.succeed(1)
      #expect(try await second.value == 1)
    }

    @Test("Unloading clears a reported failure")
    func unloadClearsFailure() async throws {
      let first = Task { try await loader.container() }
      await stub.waitForStart(1)
      stub.fail(0)
      await #expect(throws: (any Error).self) { try await first.value }
      loader.unload()
      #expect(loader.state == .notLoaded)
    }
  }
#endif
