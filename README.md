# AnyEmbeddingModel

AnyEmbeddingModel is a Swift package for text embeddings
with a choice of backend.
Each backend implements the same `EmbeddingModel` protocol
for queries and documents:

```swift
import AnyEmbeddingModel

guard let model = NaturalLanguageEmbeddingModel(language: .english) else {
    fatalError("The system has no English sentence embedding.")
}

let documents = try await model.embed(
    ["My sister lives in Lisbon.", "The printer is out of toner."],
    for: .document
)
let query = try await model.embed("Where does my sister live?", for: .query)

// Vectors are unit length, so the dot product is the cosine similarity.
let similarities = documents.map { zip($0, query).reduce(0) { $0 + $1.0 * $1.1 } }
```

Four backends are available:

- [NaturalLanguage](#naturallanguage) uses Apple's system sentence embeddings.
- [MLX](#mlx) runs models from the Hugging Face Hub on the GPU.
- [llama.cpp (GGUF)](#llamacpp-gguf) runs GGUF model files with llama.cpp.
- [OpenAI](#openai) calls servers that accept the `/v1/embeddings` format.

## Requirements

- Swift 6.3+ (Xcode 26.4+)
- macOS 14, Mac Catalyst 17, iOS 17, tvOS 17, watchOS 10, or visionOS 1 or later
- For MLX, Apple silicon and the Metal toolchain; watchOS is not supported.
- For llama.cpp, macOS or an iOS device;
  the iOS Simulator, Mac Catalyst, tvOS, watchOS, and visionOS are not supported.

## Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyEmbeddingModel.git", from: "0.1.0")
]
```

Without traits, the package provides the NaturalLanguage and OpenAI backends,
and its only dependency is Apple's
[swift-http-types](https://github.com/apple/swift-http-types).
Each trait adds a feature and its dependencies:

| Trait | Feature | Dependencies |
| --- | --- | --- |
| `MLX` | `MLXEmbeddingModel` | mlx-swift-lm, swift-transformers, swift-huggingface |
| `Llama` | `LlamaEmbeddingModel` | llama.swift |
| `AsyncHTTPClient` | AsyncHTTPClient as the default transport for `OpenAIEmbeddingModel` | async-http-client |

```swift
.package(
    url: "https://github.com/mattt/AnyEmbeddingModel.git",
    from: "0.1.0",
    traits: ["MLX"]
)
```

If your Xcode project cannot enable a package trait directly,
create a local package that enables the trait
and add that package to your project.
If dependency resolution fails with
"exhausted attempts to resolve the dependencies graph,"
add the trait's dependencies to your package as well.

## Usage

`embed(_:for:)` accepts one text or an array of texts.
It returns one `[Float]` vector per text, in input order,
with `dimension` elements and a length of 1.
Use the same model for queries and documents.

Pass plain text and a purpose, `.query` or `.document`.
The backend adds the configured prefix for that purpose.
MLX presets configure their own prefixes;
for other models, including GGUF files, set `prefixes` to match the model.
`EmbeddingPrefixes` provides `.bge`, `.e5`, `.nomic`, and `.qwen3(task:)`,
or you can supply your own query and document strings.

Store `identifier` with saved vectors,
and embed the texts again when it changes.
The identifier includes the model's revision and settings that affect vectors.
For remote models and local MLX and GGUF files,
you must update the revision or version when the model changes,
as described below.

## Backends

### NaturalLanguage

`NaturalLanguageEmbeddingModel` uses the operating system's sentence embeddings,
so it needs no model download.
Its initializer returns `nil`
if the system has no sentence embedding for the requested language or revision.

```swift
let model = NaturalLanguageEmbeddingModel(language: .english)
```

By default, the model uses the current system revision.
Pass `revision:` to select a supported revision across system updates.

### MLX

`MLXEmbeddingModel` loads models through
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)
and downloads them from the Hugging Face Hub.
It requires the `MLX` trait.
The presets configure the dimension and prefixes and pin a model commit:

```swift
let model = MLXEmbeddingModel.bgeSmallEnglish

// Optional: download and load the model before the first embedding.
try await model.prewarm()

let vectors = try await model.embed(["I like tea."], for: .document)
```

The presets are `bgeSmallEnglish`, `allMiniLM`, `nomicEmbedText`,
and `qwen3Embedding(task:)`.
`embed` loads the model if needed, so `prewarm()` is optional.
A preset keeps its pinned commit until a package update changes it.
A new commit changes the identifier.

For another model, pass its dimension, prefixes,
and a full, lowercase, 40-character commit hash.
The initializer rejects branch and tag names:

```swift
let model = try MLXEmbeddingModel(
    modelId: "BAAI/bge-small-en-v1.5",
    commit: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension: 384,
    prefixes: .bge
)
```

To start from a branch or tag, resolve its current commit through the Hub:

```swift
let model = try await MLXEmbeddingModel.resolving(
    modelId: "BAAI/bge-small-en-v1.5",
    revision: "main",
    dimension: 384,
    prefixes: .bge
)
print(model.revision)  // A full commit hash.
```

Resolving a branch or tag requires the network.
Save the returned `revision` and pass it as `commit:`
to avoid that lookup next time.
Loading still requires the model files.
If the branch moves, a new call to `resolving` returns a different identifier.

Copies of a model share one loaded copy of its weights.
Call `removeFromCache()` to release a model's weights,
or `MLXEmbeddingModel.removeAllFromCache()` to release every model's weights.
The next embedding loads them again.

While a model loads or embeds,
MLX keeps freed GPU buffers in a cache for reuse, up to a limit.
Pass `gpuMemory:` to set the limits for active and idle periods.
The default, `.automatic`, scales the active limit with the device's memory
and keeps the idle limit small;
`.unconstrained` leaves the cache effectively unlimited.
`removeFromCache()` also clears the cache when no model is loading or embedding,
unless the configuration says not to.

To load files from disk, use `init(directory:name:version:dimension:)`.
Change `version` whenever the files change,
and do not change files while that version is in use.

#### Metal library

MLX needs a compiled Metal library at runtime.
Without it, MLX stops with "Failed to load the default metallib."
Xcode builds the shaders automatically.
For an executable built with SwiftPM, use the Swift Build system,
which puts the library in `mlx-swift_Cmlx.bundle` beside the executable.
If your toolchain uses the native build system by default,
pass `--build-system swiftbuild` to `swift build`.
See [Development](#development) for the extra step needed by tests.

### llama.cpp (GGUF)

`LlamaEmbeddingModel` runs a GGUF embedding model in process with
[llama.cpp](https://github.com/ggml-org/llama.cpp),
through [llama.swift](https://github.com/mattt/llama.swift).
It requires the `Llama` trait:

```swift
let model = LlamaEmbeddingModel(
    modelPath: "/path/to/bge-small-en-v1.5-q8_0.gguf",
    name: "bge-small-en-v1.5-q8_0",
    version: "1",
    dimension: 384,
    prefixes: .bge
)

// Optional: load the model and check its dimension before the first embedding.
try await model.prewarm()

let vectors = try await model.embed(["I like tea."], for: .document)
```

The model loads a local file;
download the file first, for example from the Hugging Face Hub.
The tests use `bge-small-en-v1.5-q8_0.gguf`
from [ggml-org/bge-small-en-v1.5-Q8_0-GGUF](https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF).
The identifier uses `name` and `version`, not the path.
Change `version` whenever the file changes.

llama.cpp pools each text with the pooling type in the file's metadata.
The model accepts mean, CLS, and last-token pooling.
Other files fail to load with `LlamaEmbeddingModelError.notEmbeddingModel`:
a model that generates text has no pooling type,
and a reranker uses rank pooling, which returns a score.
GGUF files don't record a model's query and document prefixes,
so set `prefixes` to match the model.

The model puts as many texts in each call to llama.cpp
as `maximumBatchSize` and the token limit of a call allow.
By default, it offloads every layer to the GPU with Metal;
pass `gpuLayers: 0` to run on the CPU.
Copies of a model share one loaded copy of its weights.
Call `removeFromCache()` or `LlamaEmbeddingModel.removeAllFromCache()`
to release them.

ggml's Metal backend stops a process with an assertion
if the process exits while a model is still loaded on the GPU,
even when the process uses llama.cpp directly.
So `LlamaEmbeddingModel` registers an exit handler when a load ends.
The handler stops new loads and embeddings,
waits up to three seconds for running ones to stop,
and then frees every model and cached context that the package loaded.
A running embedding stops between texts while it tokenizes them
and between batches.
After the handler starts, every request throws
`EmbeddingError` with the kind `unavailable`, before any other check.

The handler is best effort.
The process can still stop with the assertion in these cases:

- It exits during the first load,
  because the handler is registered when a load ends.
- It exits from inside a load, an embedding, or the release of a model,
  for example from a llama.cpp callback.
  That work can't end while the handler waits,
  so the handler returns at once and frees nothing.
  An exit from a llama.cpp callback on a thread
  that llama.cpp or another library owns
  isn't recognized as inside that work,
  so the handler waits the full three seconds and then frees nothing.
- Work is still running after three seconds.
  The handler then frees nothing.
- It exits without running exit handlers,
  for example with `_exit` or after a crash.
- Other code loaded the models with llama.cpp,
  such as AnyLanguageModel's `LlamaLanguageModel`.

If freeing the models takes more than ten seconds,
for example because another library's log callback blocks,
the handler aborts the process, so that the exit can't hang.
A crash report for this abort shows
`LlamaLifecycle.abortAfterExitCleanupTimeout()` in its backtrace,
which first tries to write one line to standard error,
but aborts whether or not the line can be written.
If the cleanup ends just after the limit,
the cleanup's own thread can abort first,
without the line,
and the backtrace shows `LlamaLifecycle.abortAfterLateExitCleanup()`.

To be certain that no model is loaded when the process exits,
finish every load, `prewarm()`, and embedding,
start no new work, and then call `LlamaEmbeddingModel.removeAllFromCache()`.
The call cancels a running load without waiting for it,
and that load can still finish and cache its model.

To be certain that the assertion can't happen,
set the environment variable `GGML_METAL_NO_RESIDENCY` to any value
before the first model loads.
llama.cpp's Metal backend then doesn't use residency sets,
so a process exits normally even with a model loaded on the GPU.
Set it in the process's environment,
or call `setenv("GGML_METAL_NO_RESIDENCY", "1", 1)`
before the first embedding or `prewarm()`,
because `LlamaEmbeddingModel` starts the llama.cpp backend at the first load.
If another library in the process uses llama.cpp, set it before that library loads a model.
The cost is that residency sets keep a model's GPU memory wired between requests;
without them, the system can reclaim that memory while the GPU is idle,
and the next request waits for it again.
This package doesn't set the variable.

llama.cpp writes warnings and errors to standard error.
Set `LlamaEmbeddingModel.logLevel` to change the level.
llama.cpp has one log callback for the whole process.
The package installs its callback once, before its first load,
and the level applies to every llama.cpp message after that.
AnyLanguageModel's `LlamaLanguageModel` installs its own callback
when you set its `logLevel`,
and the callback installed last decides which messages appear.

The llama.cpp binary framework has slices only for macOS and iOS devices,
so the `Llama` trait doesn't build for the iOS Simulator,
Mac Catalyst, tvOS, watchOS, or visionOS.

### OpenAI

`OpenAIEmbeddingModel` sends `POST {baseURL}/embeddings`
using the OpenAI embeddings request format:

```swift
import AnyEmbeddingModel
import Foundation

let model = OpenAIEmbeddingModel(
    baseURL: URL(string: "http://localhost:8080/v1")!,
    apiKey: "",
    model: "BAAI/bge-small-en-v1.5",
    dimension: 384,
    prefixes: .bge
)
```

`baseURL` defaults to `OpenAIEmbeddingModel.defaultBaseURL`, OpenAI's API.
`apiKey` takes a `String` expression, such as a string literal or a function call.
The parameter is an autoclosure,
so the model evaluates the expression again, once for each request,
and a function that returns the current key picks up a rotated key:

```swift
let model = OpenAIEmbeddingModel(
    apiKey: currentAPIKey(),  // Called again for each request.
    model: "text-embedding-3-small",
    dimension: 1536
)
```

An empty key sends no `Authorization` header.
If an error response might contain the key,
the model withholds the server's text from the error.
The backend adds `prefixes` before sending each text.
If the server supports shorter vectors,
set `requestsDimension: true` to send `dimension` in the request.

The client cannot detect when a server changes the model behind a name.
Change `revision` when this happens so that the identifier changes too.

The model sends its requests through an `HTTPTransport`,
the same protocol that AnyLanguageModel uses.
By default, `makeDefaultTransport()` returns a `URLSession`,
or an AsyncHTTPClient client when the `AsyncHTTPClient` trait is enabled.
Pass `transport:` to use another `URLSession`, an `HTTPClient`,
or your own type that conforms to `HTTPTransport`.
A transport must not follow a redirect that changes the scheme, host, or port.
The package's `URLSession` transport, and the AsyncHTTPClient client
that `makeDefaultTransport()` returns, follow no such redirect:
they return it as an error response, so the API key and the texts stay with the server.
An `HTTPClient` that you pass yourself follows redirects as its configuration says.
It never resends the body to another server,
but it can follow a redirect that drops the body, such as a 303,
to another server as a `GET` with no body and no `Authorization` header;
the model then throws an error rather than read that server's response.

If the server rejects a request, the model throws `HTTPTransportError`
with the response and its body.

## Limitations

- MLX and llama.cpp truncate text to `maximumTokenCount` without reporting it,
  or to the model's token limit if it is smaller.
  The default is 512 tokens; the `allMiniLM` preset uses 256.
  The package does not split long texts.
- The OpenAI backend sends text without truncation;
  the server determines how to handle long input.
- The MLX and llama.cpp backends do not produce shorter vectors from Matryoshka models.
- The llama.cpp backend loads only local files,
  and it uses the pooling type in the file.

## Development

The package's tasks use [mise](https://mise.jdx.dev),
and its tests run on macOS:

| Command | Description |
| --- | --- |
| `mise run setup` | Resolve the dependencies of the package and of every trait. |
| `mise run build` | Build the library without traits. |
| `mise run build:mlx` | Build the library with the `MLX` trait. |
| `mise run build:llama` | Build the library with the `Llama` trait. |
| `mise run build:ios` | Compile the library for iOS with `xcodebuild`. |
| `mise run lint` | Check the formatting and lint rules. |
| `mise run format` | Format the Swift sources in place. |
| `mise run test` | Run tests with a local mock server and stub loads, without MLX or model downloads. |
| `mise run test:mlx` | Enable MLX and run tests with a local server in place of the Hub, without loading real models. |
| `mise run test:hub` | Load real models and compare them with reference vectors in `Tests/AnyEmbeddingModelTests/Fixtures`. |
| `mise run test:llama` | Enable llama.cpp and run the tests, with a GGUF model if `LLAMA_EMBEDDING_MODEL_PATH` is set. |

The build and test tasks read `Package.resolved` without changing it,
because a resolution without a trait would drop that trait's dependencies from the file.
After you change a dependency, run `mise run setup` to update the file.

To run the tests with AsyncHTTPClient as the default transport,
run `mise run test -- --traits AsyncHTTPClient`.

The MLX tasks look for the Metal compiler in the downloaded Metal toolchain
when `xcrun` cannot find it,
and stop with the command that downloads it if it is missing.
The `test:mlx` task uses Swift Build and has been run with Swift 6.4.
It copies `default.metallib` from the MLX bundle
beside the test binary as `mlx.metallib`, where the test process can find it.
It removes the copy after the run
so that it does not invalidate the test bundle's signature on the next build.
The task stops with an error if the toolchain cannot produce the library.
Tests that start MLX's Metal device run only in this task,
because MLX stops a process that cannot find the library.

The llama.cpp tests that load a model run only when
`LLAMA_EMBEDDING_MODEL_PATH` is the path of `bge-small-en-v1.5-q8_0.gguf`
from [ggml-org/bge-small-en-v1.5-Q8_0-GGUF](https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF).
They compare its tokens and vectors with the sentence-transformers reference for `bge-small`.
One more test runs when `LLAMA_MODEL_PATH`,
the variable that AnyLanguageModel's tests read,
is the path of a GGUF model that generates text,
and checks that the model fails to load.

The MLX real-model tests download model files as needed.
One test also asks the Hub whether each preset's commit exists.
`mise run generate:fixtures` regenerates the reference vectors
with pinned package versions.

## License

AnyEmbeddingModel is available under the Apache License 2.0.
See [LICENSE](LICENSE) for details.
