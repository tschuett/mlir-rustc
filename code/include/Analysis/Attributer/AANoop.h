#include "Analysis/Attributer/AbstractElement.h"
#include "Analysis/Attributer/AbstractState.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::analysis::attributor {

struct ANoopState : public AbstractState {
  // Returns true if in a valid state.
  // When false no information provided should be used.
  bool isValidState() const override;

  // Returns true if the state is fixed and thus does not need to be updated
  // if information changes.
  bool isAtFixpoint() const override;

  // Indicates that the abstract state should converge to the optimistic state.
  // This will usually make the optimistically assumed state the known to be
  // true state.
  //
  // Returns UNCHANGED as the assumed value does not change.
  ChangeStatus indicateOptimisticFixpoint() override;

  // Indicates that the abstract state should converge to the pessimistic state.
  // This will usually revert the optimistically assumed state to the known to
  // be true state.
  //
  // Returns CHANGED as the assumed value may change.
  ChangeStatus indicatePessimisticFixpoint() override;

  /// Unique ID (due to the unique address)
  static const char ID;
};

/// An abstract interface for liveness abstract attribute.
struct AANoop : public AbstractElement {
  AANoop(const IRPosition &IRP, Attributor &A) : AbstractElement(IRP) {}

  // Returns the internal abstract state for inspection.
  StateType &getState() override { return state; };
  const StateType &getState() const override { return state; }

  void initialize(Attributor &solver) override;

  // Returns the name of the AbstractElement for debug printing.
  const std::string getName() const override { return "AANoop"; };
  // Returns the address of the ID of the AbstractElement for type comparison.
  const void *getID() const override { return &ID; };

  // Returns the human-friendly summarized assumed state as string for
  // debugging.
  const std::string getAsStr(mlir::AsmState &asmState) const override;

  /// This function should return the address of the ID of the AbstractAttribute
  const char *getIdAddr() const override { return &state.ID; }

  /// Unique ID (due to the unique address)
  static const char ID;

  ANoopState state;
};

} // namespace rust_compiler::analysis::attributor
