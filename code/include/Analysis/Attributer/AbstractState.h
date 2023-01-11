#pragma once

namespace rust_compiler::analysis::attributor {

// A result type used to indicate if a change happened. Boolean operations on
// ChangeStatus behave as though `CHANGED` is truthy.
enum class ChangeStatus {
  UNCHANGED,
  CHANGED,
};

// Base state representing assumed and known information.
struct AbstractState {
  virtual ~AbstractState() = default;

  // Returns true if in a valid state.
  // When false no information provided should be used.
  virtual bool isValidState() const = 0;

  // Returns true if the state is fixed and thus does not need to be updated
  // if information changes.
  virtual bool isAtFixpoint() const = 0;

  // Indicates that the abstract state should converge to the optimistic state.
  // This will usually make the optimistically assumed state the known to be
  // true state.
  //
  // Returns UNCHANGED as the assumed value does not change.
  virtual ChangeStatus indicateOptimisticFixpoint() = 0;

  // Indicates that the abstract state should converge to the pessimistic state.
  // This will usually revert the optimistically assumed state to the known to
  // be true state.
  //
  // Returns CHANGED as the assumed value may change.
  virtual ChangeStatus indicatePessimisticFixpoint() = 0;
};

} // namespace rust_compiler::analysis::attributor
