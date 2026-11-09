# AnyEmbeddingModel: design

AnyEmbeddingModel gives one Swift API for text embeddings,
with a choice of engine.
It follows the shape of
[AnyLanguageModel](https://github.com/huggingface/AnyLanguageModel):
one protocol, one type for each engine,
and a package trait for each engine that adds a dependency.

This document describes the API,
and explains what it takes from the NaturalLanguage framework and what it doesn't.

## The protocol

```swift
public protocol EmbeddingModel: Sendable {
  associatedtype UnavailableReason: Sendable = Never

  var identifier: String { get }
  var dimension: Int { get }
  var availability: Availability<UnavailableReason> { get }

  func prewarm() async throws
  func embed(_ texts: [String], for purpose: EmbeddingPurpose) async throws -> [[Float]]
}

public enum EmbeddingPurpose: Sendable, Hashable, CaseIterable {
  case query
  case document
}
```

An extension adds `embed(_ text: String, for:) -> [Float]` for one text,
`isAvailable`,
an empty `prewarm()`,
and `availability` of `.available` when `UnavailableReason` is `Never`.

The names come from AnyLanguageModel where it has an equivalent:
`EmbeddingModel` matches `LanguageModel`,
`Availability<UnavailableReason>` has the same cases as its `Availability`,
`prewarm()` matches its `prewarm(for:promptPrefix:)`,
and each engine is named for its engine and the protocol,
as `MLXLanguageModel` and `OpenAILanguageModel` are.

### Engines

| Type | Engine | Trait | Dependency |
| --- | --- | --- | --- |
| `NaturalLanguageEmbeddingModel` | `NLEmbedding.sentenceEmbedding(for:)` | none | none |
| `MLXEmbeddingModel` | `MLXEmbedders` from mlx-swift-lm | `MLX` | mlx-swift-lm, swift-transformers, swift-huggingface |
| `OpenAIEmbeddingModel` | an `/v1/embeddings` server | none | none |

The remote engine uses `URLSession`,
so it needs no trait.
AnyLanguageModel has an `AsyncHTTPClient` trait for its remote models;
an embedding request is one POST with no streaming,
so this package doesn't need that option yet.

With no trait enabled,
SwiftPM didn't fetch mlx-swift-lm, swift-transformers, or swift-huggingface:
the first build after the manifest declared them left `.build/checkouts` empty.

MLX needs a Metal library at run time.
The README describes what that means for a client.

### Batches

The protocol has one batch method,
and the single-text method calls it with an array of one.
A batch is where engines differ most:
MLX pads a batch into one tensor and runs the model once,
a remote engine sends one request for many texts,
and NaturalLanguage embeds one text at a time.
A protocol with only a single-text method
would make the first two engines slow for every caller.

Each engine splits a large batch into groups that it can run,
with a `maximumBatchSize` property where the size matters.
The caller doesn't need to know the limit.

### Purposes and prefixes

Each engine takes an `EmbeddingPrefixes` value,
with one string for queries and one for documents,
and puts the string before each text for that purpose.
Callers pass plain text and a purpose;
they never add a prefix themselves.
Presets cover the common conventions:
`.bge`, `.e5`, `.nomic`, and `.qwen3(task:)`.
The MLX presets, such as `MLXEmbeddingModel.nomicEmbedText`,
choose the right prefixes for their model.

A prefix is plain text and not an enum of instruction styles,
because each model family has its own wording,
and a new family needs only a new string.
Models that take a structured instruction,
such as an `instruction` field on a request,
aren't covered.
None of the three engines has one.

### Vectors and normalization

Every engine returns unit vectors:
it divides each vector by its Euclidean length
and throws `EmbeddingError.invalidVector(index:)`
for a vector of zero length or with a value that isn't finite.
A caller can use the dot product as cosine similarity,
and vectors from different engines are comparable in scale.
There is no setting to turn normalization off.
No current use of the package needs raw vectors,
and an option would make every caller check it.

Vectors are `[Float]`.
`NLEmbedding` returns `[Double]`,
but `Float` is enough to rank by cosine similarity,
and it halves the storage.

### Dimension

`dimension` is a synchronous property that each model knows before it loads,
so a store can check its schema without loading a model.
The NaturalLanguage model reads it from `NLEmbedding`.
The MLX presets know their dimension.
Every other MLX model and every remote model takes it as a required parameter.

This stays the rule.
An `async` or optional `dimension` would make every caller handle a case
that exists only for a model the caller hasn't described.
The cost is that a caller who uses an MLX model with no preset must know its dimension.
A wrong value can't go unnoticed:
`MLXEmbeddingModel.prewarm()` and every embedding
throw `EmbeddingError.dimensionMismatch(expected:actual:)`.

### Identifier

The identifier is a string with readable fields,
separated by slashes:
the engine, the model, its revision, the dimension,
and a 64-bit FNV-1a fingerprint of the other settings that change vectors.
For example,
`mlx/BAAI%2Fbge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/384/17ad23889b01fd7d`.
The fingerprint covers both prefixes,
the maximum token count,
and, for MLX, a version string for the engine's pooling code,
so that a change to the pooling changes the identifier.

Different values can't produce the same text by accident of their boundaries.
A `/` or `%` inside a field is percent-encoded,
so a model `org/m` with a revision `v/1`
differs from a model `org/m/v` with a revision `1`.
The fingerprint hashes the number of settings,
then each setting's length and bytes.
FNV-1a isn't a cryptographic hash;
the identifier protects against mistakes, not against an attacker.

The identifier covers query settings too.
A store that saves only document vectors
could keep its vectors when only the query prefix changes,
A package that doesn't know how its vectors are stored can't make that choice,
so it changes the identifier for any change,
and the cost is an unneeded re-embedding.

The remote identifier includes the model name but not the server's URL.
Two servers that serve the same model return the same vectors,
and a change of host shouldn't force a re-embedding.
A remote server can change the model behind a name, and a client can't detect that.
A caller who knows that it happened sets `revision`.

### Revisions

A store saves vectors with the identifier.
If the weights behind an identifier change,
old and new vectors mix in one index, and nothing reports it.
`identifier` is synchronous, and resolving a branch needs the network,
so `MLXEmbeddingModel` settles the commit before a value exists.
There is no way to create an MLX model that follows a branch:

| How you create the model | Its version | Network |
| --- | --- | --- |
| A preset, such as `.bgeSmallEnglish` | A commit pinned in the package | Only to download |
| `init(modelId:commit:dimension:)`, which throws for anything but a full hash | The commit you pass | Only to download |
| `resolving(modelId:revision:dimension:)`, which is `async` | The commit that the branch or tag names now | One request, then the download |
| `init(directory:name:version:dimension:)` | The version you pass | None |

The alternatives were worse.
A failable or asynchronous `identifier` would make every caller handle a failure
in a value that a store reads constantly.
An identifier computed after the first load
wouldn't exist when a store first asks for it.
An identifier with the branch name can't change when the weights do.

The cost has three parts.
A caller who wants to follow `main` must call `resolving` each time
and accept a new identifier when the branch moves,
which makes a store embed its memories again.
That is the correct result, but it isn't free.
`resolving` needs the network,
so an app that must start offline should save `revision` from the result
and pass it to `init(modelId:commit:dimension:)`.
And a preset's commit changes only with a release of this package,
so a preset doesn't pick up improvements to a model until then.
A new pin changes the identifier, so it re-embeds too.

A local directory has no commit,
and the package can't tell whether its files changed.
`version` is the caller's promise,
and it's a required parameter so that nobody can forget it.
The version is part of the key that shares loaded weights within a process,
so a new version of a directory loads its new files.

### Truncation

Each engine says what it does with a long text:

- MLX truncates the tokens to `maximumTokenCount`,
  which is 512 by default,
  and to the model's limit if that is smaller.
  The count is part of the identifier.
- The remote engine sends the text as it is.
  Text Embeddings Inference truncates or rejects a long text
  according to its own settings,
  and OpenAI rejects it;
  a rejection is an `EmbeddingError.httpStatus(_:message:)`.
- NaturalLanguage doesn't document a limit.

Reporting truncation is out of scope for 0.1.0.
No engine tells the caller that it truncated a text,
and the package doesn't split long texts.
It is meant for short texts, such as notes and passages.
A later report can be additive,
such as a new method on `MLXEmbeddingModel` that returns token counts,
so leaving it out doesn't commit the package to a source break.

### Shorter vectors

Matryoshka models, such as nomic-embed-text-v1.5,
can return shorter vectors.
For the remote engine,
`requestsDimension` sends `dimension` to a server that supports it.
For MLX, shorter vectors are out of scope for 0.1.0.
Each model has its own recipe:
[Nomic's](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
applies layer normalization before it truncates and normalizes,
so a correct option needs a recipe and a reference test for each model.
A caller who cuts the vectors after the fact gets wrong results for such a model.
A later option can be a new parameter with a default value,
which is part of the identifier only when it is set,
so it needs no source break.

### Availability and loading

`availability` uses the same pattern as AnyLanguageModel:
the NaturalLanguage and remote models use `Never`
and are always available,
and the MLX model reports `.notLoaded` or `.failedToLoad(_:)`
until its first load finishes.
The NaturalLanguage initializer returns `nil`
when the system has no sentence embedding for a language,
so an instance is always usable.
A remote model can't know whether its server is up without a request,
so it reports no reason.

`prewarm()` downloads and loads a model.
`embed` does the same if the model isn't loaded,
so a caller never has to call `prewarm()`.

### Errors

`EmbeddingError` has the errors that more than one engine throws:
empty text, no vector, the wrong number of vectors,
the wrong dimension, a vector that can't be normalized,
an HTTP status, and a response that can't be read.
Engines also throw the errors of their frameworks,
such as `URLError` and the load errors of MLX,
without wrapping them.

Every engine rejects whitespace-only text with `EmbeddingError.emptyText(index:)`
before it embeds anything,
because the engines disagree about it:
NaturalLanguage returns `nil`,
and a transformer returns the vector of its special tokens.
A batch fails as a whole;
there are no partial results.

### Reference checks

`scripts/generate-reference-fixtures.py` records the token IDs and unit vectors
that sentence-transformers produces at a pinned commit,
with pinned package versions,
for each preset, both purposes, and the truncation limit.
The files are in `Tests/AnyEmbeddingModelTests/Fixtures`.
The engine must match the token IDs exactly and the vectors within these tolerances,
alone and in batches of unequal texts:

| Preset | Lowest cosine similarity | Largest error in one element | Measured |
| --- | ---: | ---: | --- |
| bge-small, MiniLM, nomic | 0.99999 | 1e-5 | cosine at least 0.9999997, error under 6e-7 |
| Qwen3 4-bit, ordinary sentences | 0.955 | 0.04 | cosine 0.960 to 0.974 |
| Qwen3 4-bit, one word repeated to the limit | 0.70 | 0.10 | cosine 0.72 to 0.94 |

The Qwen3 preset's reference is the full-precision model,
because sentence-transformers doesn't run the 4-bit weights.
The same engine with the full-precision weights reaches a cosine of at least 0.9996,
so the distance comes from quantization.

## What came from NLEmbedding, and what didn't

From `NLEmbedding`, the API keeps:

- A synchronous `dimension` that is known before the first vector.
- The `revision` as part of the model's identity.
  `NLEmbedding` lets a caller ask for a revision
  so that vectors stay the same after a system update.
  `NaturalLanguageEmbeddingModel(language:revision:)` exposes this,
  and the identifier includes the revision for every engine.
- A failable initializer for a language that the system doesn't have.

The API doesn't keep:

- `[Double]` vectors, for the storage reason above.
- Distance and neighbor methods,
  such as `distance(between:and:distanceType:)` and `neighbors(for:maximumCount:)`.
  These search a vocabulary that comes with the embedding,
  not the caller's texts,
  and a caller with unit vectors computes cosine distance with a dot product.
  Search belongs in a store.
- Word embeddings.
  `NLEmbedding.wordEmbedding(for:)` maps single words,
  which isn't the problem that a retrieval model solves.
- Per-token vectors, as `NLContextualEmbedding` returns.
  A pooled `NLContextualEmbedding` engine could join the package later,
  as a new type with no change to the protocol.
- The synchronous call.
  Every engine but NaturalLanguage must wait for a download, a GPU, or a server,
  so the protocol is `async`.
