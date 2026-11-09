# AnyEmbeddingModel

AnyEmbeddingModel gives one Swift API for text embeddings,
with a choice of engine.
Change the engine, and the code that uses it stays the same.

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

Every engine keeps the same promises:

- `embed` returns one vector for each text, in order.
- Each vector has `dimension` elements and a length of 1.
- You pass plain text and a purpose, `.query` or `.document`,
  and the engine adds any prefix that its model needs.
- `identifier` changes whenever the vectors could change,
  so you can save it with your vectors
  and embed your texts again when it changes.

## Engines

| Type | Runs on | Trait | Dependencies |
| --- | --- | --- | --- |
| `NaturalLanguageEmbeddingModel` | The system's sentence embedding | none | none |
| `OpenAIEmbeddingModel` | A server that accepts OpenAI's `/v1/embeddings` format | none | none |
| `MLXEmbeddingModel` | The GPU, with [MLX](https://github.com/ml-explore/mlx-swift-lm) | `MLX` | mlx-swift-lm, swift-transformers, swift-huggingface |

## Requirements

- Swift 6.2 or later
- macOS 14, iOS 17, tvOS 17, watchOS 10, or visionOS 1 or later
- For the `MLX` trait, a platform that mlx-swift-lm supports, which excludes watchOS

The package's tests run on macOS.

## Installation

Add the package to your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/mattt/AnyEmbeddingModel.git", from: "0.1.0")
]
```

With no traits, the package has no dependencies,
and SwiftPM doesn't download any.

### Traits

The `MLX` trait adds `MLXEmbeddingModel` and its dependencies:

```swift
dependencies: [
    .package(
        url: "https://github.com/mattt/AnyEmbeddingModel.git",
        from: "0.1.0",
        traits: ["MLX"]
    )
]
```

If your Xcode project can't enable a package trait directly,
add a small local package that depends on this one with the trait,
and depend on that package from your project.

Resolving packages with traits can fail with
"exhausted attempts to resolve the dependencies graph."
If this happens,
add the dependencies of the trait to your own package as well.

### MLX's Metal library

MLX needs a compiled Metal library at run time.
Without one, it stops with "Failed to load the default metallib."
What you need to do depends on how you build:

- **An app in Xcode.**
  [mlx-swift](https://github.com/ml-explore/mlx-swift) builds its shaders through Xcode,
  so an Xcode build needs no extra step.
- **An executable with `swift build`.**
  SwiftPM's native build system doesn't compile Metal shaders.
  The Swift Build system does,
  and the executable finds the library in the `mlx-swift_Cmlx.bundle` next to it.
  Swift Build is the default in Xcode's Swift 6.4 toolchain.
  If yours defaults to the native build system,
  try `swift build --build-system swiftbuild`.
- **Tests with `swift test`.**
  Even with Swift Build, a test process doesn't find that bundle.
  Copy `default.metallib` from the bundle next to the test binary,
  with the name `mlx.metallib`, for the test run.
  Remove the copy after the run,
  because it breaks the test bundle's code signature on the next build.
  [scripts/test-traits.sh](scripts/test-traits.sh) does this for this package.
  The script selects Swift Build, stops with an error if the toolchain can't produce the library,
  and has been run with Swift 6.4.

## Using each engine

### NaturalLanguage

`NLEmbedding`'s sentence embedding is part of the operating system,
so there is nothing to download.
Its initializer returns `nil` if the system has no sentence embedding for a language.

```swift
let model = NaturalLanguageEmbeddingModel(language: .english)
```

### MLX

The presets know their dimension and prefixes,
and they pin an exact commit of the model:

```swift
let model = MLXEmbeddingModel.bgeSmallEnglish

// Optional: download and load the model now.
try await model.prewarm()

let vectors = try await model.embed(["I like tea."], for: .document)
```

The presets are `bgeSmallEnglish`, `allMiniLM`, `nomicEmbedText`,
and `qwen3Embedding(task:)`.

For another model, pass a full commit hash and the dimension.
The initializer throws for a branch name,
because a branch can move to new weights while its name stays the same:

```swift
let model = try MLXEmbeddingModel(
    modelId: "BAAI/bge-small-en-v1.5",
    commit: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    dimension: 384,
    prefixes: .bge
)
```

To start from a branch or a tag, ask the Hub for its commit.
Save `revision` from the result to load the same model again without the network:

```swift
let model = try await MLXEmbeddingModel.resolving(
    modelId: "BAAI/bge-small-en-v1.5",
    revision: "main",
    dimension: 384,
    prefixes: .bge
)
print(model.revision)  // A full commit hash
```

If the branch moves, the next call returns a model with a different `identifier`.

### A server

`OpenAIEmbeddingModel` sends the OpenAI embeddings request format,
which Hugging Face Text Embeddings Inference and other servers also accept.
The package's tests use a local mock server.

```swift
import AnyEmbeddingModel
import Foundation

let model = OpenAIEmbeddingModel(
    baseURL: URL(string: "http://localhost:8080/v1")!,
    model: "BAAI/bge-small-en-v1.5",
    dimension: 384,
    prefixes: .bge
)
```

To use a hosted service, pass `apiKey`.
The format has no field for the purpose of a text,
so the model adds `prefixes` to each text before it sends the request.

A client can't tell when a server changes the model behind a name.
If you know that it happened, change the `revision` parameter,
which is part of the identifier.

## Using a model with Recall

[Recall](https://github.com/mattt/Recall) is a local memory store for Swift.
Neither package depends on the other.
A short closure connects them:

```swift
import AnyEmbeddingModel
import Foundation
import Recall

let model = MLXEmbeddingModel.bgeSmallEnglish
let embedder = Embedder(identifier: model.identifier) { text, purpose in
    try await model.embed([text], for: purpose == .query ? .query : .document)[0]
}
let url = URL.documentsDirectory.appending(path: "memories.sqlite")
let store = try MemoryStore(url: url, embedder: embedder)
```

## Limits

- Long texts: the MLX engine truncates a text to `maximumTokenCount` without an error,
  and the package doesn't split long texts.
  It is meant for short texts, such as notes and passages.
- Shorter vectors: for MLX, the package doesn't produce shorter vectors from Matryoshka models.

[docs/design.md](docs/design.md) explains the design and these choices.

## Testing

- `swift test` needs neither MLX nor the network.
  It uses a local mock server and stub loads.
- `scripts/test-traits.sh` adds the MLX code that needs no real model,
  with a local server in place of the Hub.
- `ANY_EMBEDDING_MODEL_TEST_HUB=1 scripts/test-traits.sh` also loads real models
  and checks them against the reference vectors in `Tests/AnyEmbeddingModelTests/Fixtures`.
  It downloads the models on the first run,
  and it can run from the cache after that,
  except for one test that asks the Hub whether each preset's commit exists.

`scripts/generate-reference-fixtures.py` regenerates the reference vectors,
with `uv run` and pinned package versions.

## License

AnyEmbeddingModel is available under the MIT license.
See [LICENSE.md](LICENSE.md) for details.
