# /// script
# requires-python = ">=3.12"
# ///
"""Prints the expected identifiers that the identifier tests compare against.

This is an independent implementation of the format in
Sources/AnyEmbeddingModel/Shared/Identifier.swift.

Usage: uv run scripts/generate-identifier-goldens.py
"""

import struct
import urllib.parse

OFFSET = 0xCBF29CE484222325
PRIME = 0x100000001B3
MASK = (1 << 64) - 1
QWEN3_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def fingerprint(settings):
    state = OFFSET

    def add(data):
        nonlocal state
        for byte in data:
            state = ((state ^ byte) * PRIME) & MASK

    add(struct.pack("<Q", len(settings)))
    for setting in settings:
        encoded = setting.encode("utf-8")
        add(struct.pack("<Q", len(encoded)))
        add(encoded)
    return f"{state:016x}"


def encode(field):
    # quote() leaves letters, digits, and "_.-~" unchanged.
    return urllib.parse.quote(field.encode("utf-8"), safe="")


def identifier(engine, model, revision, dimension, settings):
    if revision is None:
        revision_field = "-"
    elif revision == "-":
        revision_field = "%2D"
    else:
        revision_field = encode(revision)
    fields = [
        encode(engine),
        encode(model),
        revision_field,
        str(dimension),
        fingerprint(settings),
    ]
    return "/".join(fields)


def main():
    pool = "pool-v2"
    bge = "Represent this sentence for searching relevant passages: "
    cases = {
        "mlx bge-small": (
            "mlx",
            "BAAI/bge-small-en-v1.5",
            "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            384,
            [bge, "", "512", pool],
        ),
        "mlx minilm": (
            "mlx",
            "sentence-transformers/all-MiniLM-L6-v2",
            "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            384,
            ["", "", "256", pool],
        ),
        "mlx nomic": (
            "mlx",
            "nomic-ai/nomic-embed-text-v1.5",
            "e9b6763023c676ca8431644204f50c2b100d9aab",
            768,
            ["search_query: ", "search_document: ", "512", pool],
        ),
        "mlx qwen3": (
            "mlx",
            "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
            "6c3ae70858513f1a78e9cdca3cae330d9075cd2a",
            1024,
            [f"Instruct: {QWEN3_TASK}\nQuery:", "", "512", pool],
        ),
        "mlx local": ("mlx", "my model", "v/1", 3, ["", "", "512", pool]),
        "naturallanguage": ("naturallanguage", "sentence-en", "1", 512, ["", ""]),
        "openai plain": ("openai", "text-embedding-3-small", None, 1536, ["", "", ""]),
        "openai tag": (
            "openai",
            "nomic-embed-text:latest",
            "2",
            768,
            ["search_query: ", "search_document: ", "dimensions"],
        ),
        "openai non-ascii": ("openai", "modèle", None, 3, ["", "", ""]),
        "openai decomposed": ("openai", "modèle", None, 3, ["", "", ""]),
    }
    for name, case in cases.items():
        print(f"{name}: {identifier(*case)}")


if __name__ == "__main__":
    main()
