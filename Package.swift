// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "AnyEmbeddingModel",
    platforms: [
        .macOS(.v14),
        .macCatalyst(.v17),
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
        .trait(
            name: "AsyncHTTPClient",
            description: "Sends OpenAIEmbeddingModel's requests with AsyncHTTPClient by default."
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
        .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.26.0"),
        .package(url: "https://github.com/apple/swift-http-types", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "AnyEmbeddingModel",
            dependencies: [
                .product(name: "HTTPTypes", package: "swift-http-types"),
                .product(
                    name: "HTTPTypesFoundation",
                    package: "swift-http-types",
                    // It extends URLSession, which WASI doesn't have.
                    condition: .when(platforms: [
                        .macOS, .macCatalyst, .iOS, .tvOS, .watchOS, .visionOS, .linux, .windows, .android,
                    ])
                ),
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
                .product(
                    name: "AsyncHTTPClient",
                    package: "async-http-client",
                    condition: .when(traits: ["AsyncHTTPClient"])
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
                .product(
                    name: "AsyncHTTPClient",
                    package: "async-http-client",
                    condition: .when(traits: ["AsyncHTTPClient"])
                ),
            ],
            exclude: ["Fixtures"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
