#pragma once

#include "Analysis/Attributer/AbstractState.h"
#include "Analysis/Attributer/DependencyGraphNode.h"
#include "Analysis/Attributer/IRPosition.h"

#include <mlir/IR/AsmState.h>

namespace rust_compiler::analysis::attributor {

class Attributor;

// Base type for information in the solver framework.
// Each element represents some assumed and known knowledge anchored on a
// specific position in the IR such as a Value or Operation.
class AbstractElement : public IRPosition, public DependencyGraphNode {
public:
  using StateType = AbstractState;

  AbstractElement(const IRPosition &pos) : IRPosition(pos) {}
  virtual ~AbstractElement() = default;

  // Returns an IR position anchoring this element to the IR.
  const IRPosition &getPosition() const { return *this; };
  IRPosition &getPosition() { return *this; };

  // Returns the internal abstract state for inspection.
  virtual StateType &getState() = 0;
  virtual const StateType &getState() const = 0;

  /// Return an IR position, see struct IRPosition.
  const IRPosition &getIRPosition() const { return *this; };
  IRPosition &getIRPosition() { return *this; };

  // Initializes the state with the information in |solver|.
  //
  // This function is called by the solver once all abstract elements
  // have been identified. It can and shall be used for tasks like:
  //  - identify existing knowledge in the IR and use it for the "known state"
  //  - perform any work that is not going to change over time, e.g., determine
  //    a subset of the IR, or elements in-flight, that have to be looked at
  //    in the `updateImpl` method.
  virtual void initialize(Attributor &solver) {}

  // Returns the name of the AbstractElement for debug printing.
  virtual const std::string getName() const = 0;
  // Returns the address of the ID of the AbstractElement for type comparison.
  virtual const void *getID() const = 0;

  // Returns true if |node| is of type AbstractElement so that the dyn_cast and
  // cast can use such information to cast an DepGraphNode to an
  // AbstractElement.
  //
  // We eagerly return true here because all DepGraphNodes except for the
  // synthethis node are of type AbstractElement.
  static bool classof(const DependencyGraphNode *node) { return true; }

  // Returns the human-friendly summarized assumed state as string for
  // debugging.
  virtual const std::string getAsStr(mlir::AsmState &asmState) const = 0;

  void print(llvm::raw_ostream &os, mlir::AsmState &asmState) const override;
  virtual void printWithDeps(llvm::raw_ostream &os,
                             mlir::AsmState &asmState) const;
  void dump(mlir::AsmState &asmState) const;

  friend class Attributor;

  /// This function should return the address of the ID of the AbstractAttribute
  virtual const char *getIdAddr() const = 0;

protected:
  // Hook for the solver to trigger an update of the internal state.
  //
  // If this attribute is already fixed this method will return UNCHANGED,
  // otherwise it delegates to `AbstractElement::updateImpl`.
  //
  // Returns CHANGED if the internal state changed, otherwise UNCHANGED.
  ChangeStatus update(Attributor &solver);

  // Update/transfer function which has to be implemented by the derived
  // classes.
  //
  // When called the environment has changed and we have to determine if
  // the current information is still valid or adjust it otherwise.
  //
  // Returns CHANGED if the internal state changed, otherwise UNCHANGED.
  virtual ChangeStatus updateImpl(Attributor &solver) = 0;
};

} // namespace rust_compiler::analysis::attributor
