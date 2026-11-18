#if Llama
    import Foundation
    import Testing

    @testable import AnyEmbeddingModel

    /// Tests of the exit handler's bookkeeping, without a model.
    @Suite("LlamaLifecycle")
    struct LlamaLifecycleTests {
        /// Starts work on another thread that runs until the returned semaphore is signaled,
        /// and returns once the work is running.
        static func startHeldWork(_ lifecycle: LlamaLifecycle) -> DispatchSemaphore {
            let started = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            Thread.detachNewThread {
                try? lifecycle.perform {
                    started.signal()
                    release.wait()
                }
            }
            started.wait()
            return release
        }

        @Test("Shutting down with no work frees, and a second shutdown does nothing")
        func shutDownIdle() {
            let lifecycle = LlamaLifecycle()
            #expect(lifecycle.shutDown(deadline: 0) == .freed)
            #expect(lifecycle.shutDown(deadline: 0) == .alreadyShutDown)
            #expect(lifecycle.isStopping)
            #expect(throws: EmbeddingError.self) { try lifecycle.perform {} }
        }

        @Test("Shutting down waits for running work, and frees nothing at the deadline")
        func shutDownTimesOut() async {
            let lifecycle = LlamaLifecycle()
            let release = Self.startHeldWork(lifecycle)
            // Shut down from another thread, so that the work isn't its own.
            let result = await withCheckedContinuation { continuation in
                Thread.detachNewThread {
                    let start = ContinuousClock.now
                    let result = lifecycle.shutDown(deadline: 0.2)
                    continuation.resume(returning: (result, ContinuousClock.now - start))
                }
            }
            #expect(result.0 == .timedOut)
            #expect(result.1 >= .milliseconds(200))
            release.signal()
        }

        @Test("Shutting down returns when running work ends")
        func shutDownWaitsForWork() async {
            let lifecycle = LlamaLifecycle()
            let release = Self.startHeldWork(lifecycle)
            let result = await withCheckedContinuation { continuation in
                Thread.detachNewThread {
                    continuation.resume(returning: lifecycle.shutDown(deadline: 10))
                }
                usleep(100_000)
                release.signal()
            }
            #expect(result == .freed)
        }

        @Test("Shutting down inside work on the current thread returns at once")
        func shutDownInsideThreadOperation() throws {
            let lifecycle = LlamaLifecycle()
            // The handler returns this result before its wait.
            let result = try lifecycle.perform { lifecycle.shutDown(deadline: 10) }
            #expect(result == .insideOperation)
        }

        @Test("The cleanup deadline does nothing after the cleanup ends")
        func cleanupDeadlineAfterCleanup() async {
            let name = "cleanupDeadlineAfterCleanup"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .success) {
                ExitTestTimer.childStarted("cleanupDeadlineAfterCleanup")
                ExitTestTimer.stepStarted("cleanupDeadlineAfterCleanup")
                let value = LlamaLifecycle.withAbortDeadline(0.1) { 42 }
                // Outlive the deadline; the watchdog must not abort.
                usleep(500_000)
                exit(value == 42 ? 0 : 5)
            }
            await ExitTestTimer.finish(name, watchdog, measuring: false)
        }

        @Test("A cleanup that runs past its deadline aborts the process")
        func cleanupDeadlinePassed() async throws {
            let name = "cleanupDeadlinePassed"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .signal(SIGABRT)) {
                ExitTestTimer.childStarted("cleanupDeadlinePassed")
                ExitTestTimer.stepStarted("cleanupDeadlinePassed")
                LlamaLifecycle.withAbortDeadline(0.2) {
                    while true { sleep(60) }
                }
            }
            let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
            #expect(elapsed >= .milliseconds(200))
        }

        @Test("A cleanup that runs past its deadline aborts even if standard error is a full pipe")
        func cleanupDeadlineWithFullStandardError() async throws {
            let name = "cleanupDeadlineWithFullStandardError"
            let watchdog = ExitTestTimer.watch(name)
            await #expect(processExitsWith: .signal(SIGABRT)) {
                ExitTestTimer.childStarted("cleanupDeadlineWithFullStandardError")
                // Make standard error a pipe that is full and that nobody reads.
                // The read end stays open, so a write blocks instead of failing.
                var fds: [CInt] = [0, 0]
                guard pipe(&fds) == 0 else { _exit(3) }
                let flags = fcntl(fds[1], F_GETFL)
                _ = fcntl(fds[1], F_SETFL, flags | O_NONBLOCK)
                let chunk = [UInt8](repeating: 0x2E, count: 4096)
                while chunk.withUnsafeBytes({ Darwin.write(fds[1], $0.baseAddress, $0.count) }) > 0 {}
                guard errno == EAGAIN else { _exit(4) }
                _ = fcntl(fds[1], F_SETFL, flags)
                guard dup2(fds[1], STDERR_FILENO) == STDERR_FILENO else { _exit(6) }
                ExitTestTimer.stepStarted("cleanupDeadlineWithFullStandardError")
                LlamaLifecycle.withAbortDeadline(0.2) {
                    while true { sleep(60) }
                }
            }
            let elapsed = try #require(await ExitTestTimer.finish(name, watchdog))
            #expect(elapsed >= .milliseconds(200))
        }

        #if DEBUG
            @Test("Once the cleanup completes, a watchdog that timed out doesn't abort")
            func cleanupCompletionWinsRace() async {
                let name = "cleanupCompletionWinsRace"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .success) {
                    ExitTestTimer.childStarted("cleanupCompletionWinsRace")
                    let entered = DispatchSemaphore(value: 0)
                    let proceed = DispatchSemaphore(value: 0)
                    let decided = DispatchSemaphore(value: 0)
                    let aborts = Locked<Bool?>(nil)
                    LlamaTestHooks.cleanupWatchdog.withLock {
                        $0 = { event in
                            switch event {
                            case .beforeDecision:
                                // Hold the watchdog after its timeout
                                // until the cleanup has completed.
                                entered.signal()
                                proceed.wait()
                            case .decided(let value):
                                aborts.withLock { $0 = value }
                                decided.signal()
                            }
                        }
                    }
                    ExitTestTimer.stepStarted("cleanupCompletionWinsRace")
                    // The cleanup returns only after the watchdog's deadline has passed.
                    let value = LlamaLifecycle.withAbortDeadline(0.05) {
                        entered.wait()
                        return 42
                    }
                    proceed.signal()
                    decided.wait()
                    exit(value == 42 && aborts.withLock({ $0 }) == false ? 0 : 5)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }

            @Test("Once a watchdog decides to abort, the cleanup doesn't return")
            func cleanupTimeoutWinsRace() async {
                let name = "cleanupTimeoutWinsRace"
                let watchdog = ExitTestTimer.watch(name)
                await #expect(processExitsWith: .signal(SIGABRT)) {
                    ExitTestTimer.childStarted("cleanupTimeoutWinsRace")
                    let decided = DispatchSemaphore(value: 0)
                    LlamaTestHooks.cleanupWatchdog.withLock {
                        $0 = { event in
                            guard event == .decided(aborts: true) else { return }
                            decided.signal()
                            // Keep the watchdog from aborting,
                            // so that only the cleanup's own check can abort.
                            while true { sleep(60) }
                        }
                    }
                    ExitTestTimer.stepStarted("cleanupTimeoutWinsRace")
                    _ = LlamaLifecycle.withAbortDeadline(0.05) {
                        decided.wait()
                        return 42
                    }
                    // A distinct status shows that the cleanup returned.
                    exit(5)
                }
                await ExitTestTimer.finish(name, watchdog, measuring: false)
            }
        #endif

        @Test("A failed registration of the exit handler makes the handler free nothing")
        func registrationFailure() {
            let lifecycle = LlamaLifecycle()
            lifecycle.exitHandlerRegistrationFailed()
            #expect(lifecycle.shutDown(deadline: 0) == .registrationFailed)
        }
    }

    /// Lets the parent of an exit test kill a child that hangs,
    /// and measure how long the child takes to end after the step under test.
    ///
    /// Swift Testing's time limit doesn't end a child process,
    /// so a child that hangs would keep the test run waiting.
    /// The child writes its process ID when it starts,
    /// and the monotonic time when the step under test starts,
    /// to files that both processes name from the parent's process ID.
    /// The measurement starts after the child's setup,
    /// so a slow load doesn't change it.
    ///
    /// The measurement ends when the parent sees that the child has ended,
    /// which a busy machine can delay by seconds.
    /// So a test checks only that a time is at least a bound,
    /// which a delay can't break,
    /// and never that a time is less than a bound.
    enum ExitTestTimer {
        /// How long the parent waits for a child before it kills the child.
        static let limit: Duration = .seconds(120)

        /// The current time of a monotonic clock that every process on the system shares.
        static func now() -> UInt64 {
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        }

        private static func file(_ name: String, _ suffix: String, parent: pid_t) -> URL {
            URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("llama-exit-test-\(parent)-\(name).\(suffix)")
        }

        /// The exit status of a child that couldn't write a timer file.
        ///
        /// The child also writes the error to standard error.
        static let writeFailureStatus: CInt = 70

        /// In the child: writes `text` to a timer file,
        /// or ends the child with ``writeFailureStatus`` if it can't.
        private static func write(_ text: String, to url: URL) {
            do {
                try text.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                fputs("ExitTestTimer couldn't write \(url.path): \(error)\n", stderr)
                _exit(writeFailureStatus)
            }
        }

        /// In the child: records its process ID.
        static func childStarted(_ name: String) {
            write(String(getpid()), to: file(name, "pid", parent: getppid()))
        }

        /// In the child: records when the step under test starts.
        static func stepStarted(_ name: String) {
            write(String(now()), to: file(name, "start", parent: getppid()))
        }

        private static func read(_ name: String, _ suffix: String) -> String? {
            try? String(contentsOf: file(name, suffix, parent: getpid()), encoding: .utf8)
        }

        /// In the parent: starts a task that kills the child if it hasn't ended after ``limit``,
        /// and returns whether it did.
        static func watch(_ name: String) -> Task<Bool, Never> {
            for suffix in ["pid", "start"] {
                try? FileManager.default.removeItem(at: file(name, suffix, parent: getpid()))
            }
            return Task.detached {
                guard (try? await Task.sleep(for: limit)) != nil else { return false }
                guard let pid = read(name, "pid").flatMap({ pid_t($0) }) else { return false }
                kill(pid, SIGKILL)
                return true
            }
        }

        /// In the parent, after the child ends: stops the watchdog,
        /// records an issue if it killed the child,
        /// and, if `measuring`, returns the time from the step's start to now.
        ///
        /// If the start time can't be read,
        /// this records an issue with the error and returns `nil`.
        /// A child that couldn't write the file
        /// exits with ``writeFailureStatus`` instead.
        @discardableResult
        static func finish(
            _ name: String,
            _ watchdog: Task<Bool, Never>,
            measuring: Bool = true
        ) async -> Duration? {
            watchdog.cancel()
            let killed = await watchdog.value
            #expect(!killed, "The child process didn't end within \(limit).")
            guard measuring else { return nil }
            let url = file(name, "start", parent: getpid())
            let text: String
            do {
                text = try String(contentsOf: url, encoding: .utf8)
            } catch {
                Issue.record("Couldn't read the step's start time from \(url.path): \(error)")
                return nil
            }
            guard let start = UInt64(text) else {
                Issue.record("The step's start time in \(url.path) isn't a number: \(text.debugDescription)")
                return nil
            }
            return .nanoseconds(Int64(now() - start))
        }
    }
#endif
