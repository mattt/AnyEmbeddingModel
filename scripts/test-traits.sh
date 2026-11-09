#!/bin/bash
# Builds and tests the package with its traits enabled.
#
# MLX needs a compiled Metal library at run time.
# SwiftPM's native build system doesn't compile one, so this script selects Swift Build,
# which puts the library in mlx-swift_Cmlx.bundle.
# A test process doesn't find that bundle.
# MLX also searches for mlx.metallib next to the binary,
# so this script copies the library there for the test run
# and removes it after the run.
# A copy left in the test bundle breaks its code signature on the next build.
#
# This script needs a toolchain whose SwiftPM has the Swift Build system.
# It has been run with Swift 6.4, and it stops with an error
# if the toolchain doesn't have Swift Build or doesn't produce the library.
#
# Tests that load real models run only if ANY_EMBEDDING_MODEL_TEST_HUB is set.
#
# Usage: scripts/test-traits.sh [traits] [swift test arguments...]
# The default trait is "MLX".
set -euo pipefail

cd "$(dirname "$0")/.."

traits="${1:-MLX}"
if [ "$#" -gt 0 ]; then shift; fi

if ! xcrun swift build --help | grep -q "swiftbuild"; then
  echo "error: this toolchain's SwiftPM has no Swift Build system," >&2
  echo "which this script needs to compile MLX's Metal library." >&2
  echo "Use a newer toolchain, or build and test MLX code with Xcode." >&2
  exit 1
fi

options=(--build-system swiftbuild --traits "$traits")

xcrun swift build --build-tests "${options[@]}"

products="$(xcrun swift build "${options[@]}" --show-bin-path)"
metallib="$products/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
copy="$products/AnyEmbeddingModelTests.xctest/Contents/MacOS/mlx.metallib"
if [[ "$traits" == *MLX* ]]; then
  if [ ! -f "$metallib" ]; then
    echo "error: this toolchain didn't produce MLX's Metal library: $metallib" >&2
    echo "This script has been run with Swift 6.4." >&2
    exit 1
  fi
  cp "$metallib" "$copy"
  trap 'rm -f "$copy"' EXIT
fi

xcrun swift test --skip-build "${options[@]}" "$@"
