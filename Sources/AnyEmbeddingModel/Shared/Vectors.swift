/// Operations that every engine uses on its inputs and outputs.
enum Vectors {
  /// Throws ``EmbeddingError/emptyText(index:)``
  /// for the first text that contains only whitespace.
  static func validate(_ texts: [String]) throws(EmbeddingError) {
    for (index, text) in texts.enumerated()
    where text.allSatisfy({ $0.isWhitespace || $0.isNewline }) {
      throw .emptyText(index: index)
    }
  }

  /// Returns `vector` scaled to unit length.
  ///
  /// - Throws: ``EmbeddingError/dimensionMismatch(expected:actual:)``
  ///   if the vector doesn't have `dimension` elements,
  ///   or ``EmbeddingError/invalidVector(index:)``
  ///   if it has zero length or a value that isn't finite.
  static func normalized(
    _ vector: [Float],
    dimension: Int,
    index: Int
  ) throws(EmbeddingError) -> [Float] {
    guard vector.count == dimension else {
      throw .dimensionMismatch(expected: dimension, actual: vector.count)
    }
    // Sum in Double, so that large values don't overflow before the square root.
    var sum: Double = 0
    for value in vector {
      guard value.isFinite else { throw .invalidVector(index: index) }
      sum += Double(value) * Double(value)
    }
    let length = sum.squareRoot()
    guard length.isNormal else { throw .invalidVector(index: index) }
    return vector.map { Float(Double($0) / length) }
  }

  /// Returns each vector scaled to unit length,
  /// after a check that there is one vector for each text.
  ///
  /// - Parameter firstIndex: The index of the first vector in the caller's whole input,
  ///   for errors about a vector in a batch that is part of a larger input.
  static func normalized(
    _ vectors: [[Float]],
    count: Int,
    dimension: Int,
    firstIndex: Int = 0
  ) throws(EmbeddingError) -> [[Float]] {
    guard vectors.count == count else {
      throw .wrongVectorCount(expected: count, actual: vectors.count)
    }
    var result: [[Float]] = []
    result.reserveCapacity(vectors.count)
    for (index, vector) in vectors.enumerated() {
      result.append(try normalized(vector, dimension: dimension, index: firstIndex + index))
    }
    return result
  }

  /// Splits `elements` into consecutive groups of at most `size` elements.
  static func batches<T>(_ elements: [T], size: Int) -> [ArraySlice<T>] {
    precondition(size > 0, "The batch size must be greater than zero.")
    return stride(from: 0, to: elements.count, by: size).map {
      elements[$0..<min($0 + size, elements.count)]
    }
  }
}
