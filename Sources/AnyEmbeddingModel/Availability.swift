/// Whether an embedding model can embed text now.
///
/// This type has the same shape as AnyLanguageModel's `Availability`,
/// so a `switch` over a model's availability reads the same in both packages.
public enum Availability<UnavailableReason> {
    /// The model is ready.
    case available

    /// The model isn't ready, for the reason in the associated value.
    case unavailable(UnavailableReason)
}

extension Availability: Equatable where UnavailableReason: Equatable {}
extension Availability: Hashable where UnavailableReason: Hashable {}
extension Availability: Sendable where UnavailableReason: Sendable {}
