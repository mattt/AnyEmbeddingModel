// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "AnyEmbeddingModel",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .watchOS(.v10),
    .visionOS(.v1),
  ],
  products: [
    .library(name: "AnyEmbeddingModel", targets: ["AnyEmbeddingModel"])
  ],
  traits: [
    .trait(
      name: "MLX",
      description: "Enables MLXEmbeddingModel, which runs embedding models with MLX."
    ),
    .default(enabledTraits: []),
  ],
  dependencies: [
    .package(
      url: "https://github.com/ml-explore/mlx-swift-lm",
      .upToNextMinor(from: "3.32.3"),
      traits: []
    ),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
    .package(url: "https://github.com/huggingface/swift-huggingface", from: "1.0.0"),
  ],
  targets: [
    .target(
      name: "AnyEmbeddingModel",
      dependencies: [
        .product(
          name: "MLXEmbedders",
          package: "mlx-swift-lm",
          condition: .when(traits: ["MLX"])
        ),
        .product(
          name: "MLXLMCommon",
          package: "mlx-swift-lm",
          condition: .when(traits: ["MLX"])
        ),
        .product(
          name: "MLXHuggingFace",
          package: "mlx-swift-lm",
          condition: .when(traits: ["MLX"])
        ),
        .product(
          name: "HuggingFace",
          package: "swift-huggingface",
          condition: .when(traits: ["MLX"])
        ),
        .product(
          name: "Tokenizers",
          package: "swift-transformers",
          condition: .when(traits: ["MLX"])
        ),
      ]
    ),
    .testTarget(
      name: "AnyEmbeddingModelTests",
      dependencies: [
        "AnyEmbeddingModel",
        .product(
          name: "HuggingFace",
          package: "swift-huggingface",
          condition: .when(traits: ["MLX"])
        ),
      ],
      exclude: ["Fixtures"]
    ),
  ],
  swiftLanguageModes: [.v6]
)
