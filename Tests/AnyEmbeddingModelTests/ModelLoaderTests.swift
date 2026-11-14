#if canImport(Network)
    import Foundation
    import Testing

    @testable import AnyEmbeddingModel

    /// An error for a wait in a test that didn't finish in time.
    private struct WaitTimedOut: Error, CustomStringConvertible {
        let description: String
    }

    /// Waits for a condition, and throws if it doesn't become true in time.
    private func waitUntil(
        _ description: String,
        _ condition: () -> Bool
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(10))
        while !condition() {
            guard clock.now < deadline else { throw WaitTimedOut(description: description) }
            await Task.yield()
        }
    }

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

        func waitForStart(_ count: Int) async throws {
            try await waitUntil("load \(count) to start") { started >= count }
        }

        func succeed(_ index: Int) {
            take(index)?.resume(returning: index)
        }

        func fail(_ index: Int) {
            take(index)?.resume(throwing: Failure())
        }

        /// Ends every load that a test left running, so that no task waits forever.
        func completeRemaining() {
            let remaining = continuations.withLock { continuations in
                defer { continuations = continuations.map { _ in nil } }
                return continuations.compactMap { $0 }
            }
            for continuation in remaining {
                continuation.resume(throwing: Failure())
            }
        }

        private func take(_ index: Int) -> CheckedContinuation<Int, any Error>? {
            continuations.withLock { continuations in
                defer { continuations[index] = nil }
                return continuations[index]
            }
        }
    }

    /// A caller of a loader, with a result that a test can wait for with a limit.
    private final class Caller: Sendable {
        private let result = TestLock<Result<Int, any Error>?>(nil)
        private let task: Task<Void, Never>

        init(_ loader: ModelLoader<Int>) {
            let result = self.result
            task = Task {
                let outcome = await Task { try await loader.container() }.result
                result.withLock { $0 = outcome }
            }
        }

        private func outcome() async throws -> Result<Int, any Error> {
            try await waitUntil("a caller to finish") { result.withLock { $0 != nil } }
            guard let outcome = result.withLock({ $0 }) else {
                throw WaitTimedOut(description: "a caller's result")
            }
            return outcome
        }

        func value() async throws -> Int {
            try await outcome().get()
        }

        func failed() async throws -> Bool {
            if case .failure = try await outcome() { return true }
            return false
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

        /// Runs a test, and ends every started load afterward, even if the test fails.
        private func run(_ body: () async throws -> Void) async throws {
            defer { stub.completeRemaining() }
            try await body()
        }

        /// Starts callers, and returns after all of them have joined a load
        /// that has `load + 1` starts.
        private func join(_ count: Int, load: Int = 0, alreadyJoined: Int = 0) async throws -> [Caller] {
            let callers = (0 ..< count).map { _ in Caller(loader) }
            try await stub.waitForStart(load + 1)
            let expected = count + alreadyJoined
            try await waitUntil("\(expected) callers to join") { loader.waiterCount == expected }
            return callers
        }

        @Test("Several waiters share one load")
        func waitersShareALoad() async throws {
            try await run {
                let callers = try await join(5)
                stub.succeed(0)
                for caller in callers {
                    #expect(try await caller.value() == 0)
                }
                #expect(stub.started == 1)
                #expect(loader.state == .loaded)
            }
        }

        @Test("A loaded value is reused")
        func reuse() async throws {
            try await run {
                let first = try await join(1)
                stub.succeed(0)
                _ = try await first[0].value()
                #expect(try await loader.container() == 0)
                #expect(stub.started == 1)
            }
        }

        @Test("Unloading during a load leaves the loader unloaded when the load ends")
        func unloadDuringLoad() async throws {
            try await run {
                let caller = try await join(1)
                loader.unload()
                #expect(loader.state == .notLoaded)

                stub.succeed(0)
                // The caller still gets the result of the load that it waited for.
                #expect(try await caller[0].value() == 0)
                #expect(loader.state == .notLoaded)
            }
        }

        @Test("An abandoned failure doesn't report a failure or erase a newer load")
        func abandonedFailure() async throws {
            try await run {
                let abandoned = try await join(1)
                loader.unload()
                let current = try await join(1, load: 1)

                stub.fail(0)
                #expect(try await abandoned[0].failed())
                #expect(loader.state == .notLoaded)

                // A new caller joins the current load and doesn't start a third.
                let joined = try await join(1, load: 1, alreadyJoined: 1)
                stub.succeed(1)
                #expect(try await current[0].value() == 1)
                #expect(try await joined[0].value() == 1)
                #expect(stub.started == 2)
                #expect(loader.state == .loaded)
            }
        }

        @Test("An abandoned success doesn't replace a newer value")
        func abandonedSuccess() async throws {
            try await run {
                let abandoned = try await join(1)
                loader.unload()
                let current = try await join(1, load: 1)

                stub.succeed(1)
                #expect(try await current[0].value() == 1)

                stub.succeed(0)
                #expect(try await abandoned[0].value() == 0)
                #expect(try await loader.container() == 1)
                #expect(stub.started == 2)
            }
        }

        @Test("A failure is reported, and the next call tries again")
        func retryAfterFailure() async throws {
            try await run {
                let first = try await join(1)
                stub.fail(0)
                #expect(try await first[0].failed())
                #expect(loader.state == .failed("stub failure"))

                let second = try await join(1, load: 1)
                #expect(loader.state == .notLoaded)
                stub.succeed(1)
                #expect(try await second[0].value() == 1)
                #expect(loader.state == .loaded)
            }
        }

        @Test("All waiters on a failed load fail, and the state reports one failure")
        func waitersOnAFailure() async throws {
            try await run {
                let callers = try await join(3)
                stub.fail(0)
                for caller in callers {
                    #expect(try await caller.failed())
                }
                #expect(stub.started == 1)
                #expect(loader.state == .failed("stub failure"))
            }
        }

        @Test("Unloading a loaded value makes the next call load again")
        func unloadThenReload() async throws {
            try await run {
                let first = try await join(1)
                stub.succeed(0)
                _ = try await first[0].value()

                loader.unload()
                #expect(loader.state == .notLoaded)

                let second = try await join(1, load: 1)
                stub.succeed(1)
                #expect(try await second[0].value() == 1)
            }
        }

        @Test("Unloading clears a reported failure")
        func unloadClearsFailure() async throws {
            try await run {
                let first = try await join(1)
                stub.fail(0)
                #expect(try await first[0].failed())
                loader.unload()
                #expect(loader.state == .notLoaded)
            }
        }
    }
#endif
