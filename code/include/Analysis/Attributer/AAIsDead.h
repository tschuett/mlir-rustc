#include "Analysis/Attributer/AbstractElement.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace rust_compiler::analysis::attributor {

/// An abstract interface for liveness abstract attribute.
struct AAIsDead : public AbstractElement {
  AAIsDead(const IRPosition &IRP, Attributor &A) : AbstractElement(IRP) {}

  /// State encoding bits. A set bit in the state means the property holds.
  enum {
    HAS_NO_EFFECT = 1 << 0,
    IS_REMOVABLE = 1 << 1,

    IS_DEAD = HAS_NO_EFFECT | IS_REMOVABLE,
  };
  //static_assert(IS_DEAD == getBestState(), "Unexpected BEST_STATE value");

protected:
  /// The query functions are protected such that other attributes need to go
  /// through the Attributor interfaces: `Attributor::isAssumedDead(...)`

  /// Returns true if the underlying value is assumed dead.
  virtual bool isAssumedDead() const = 0;

  /// Returns true if the underlying value is known dead.
  virtual bool isKnownDead() const = 0;

  /// Returns true if \p BB is assumed dead.
  virtual bool isAssumedDead(const mlir::Block *BB) const = 0;

  /// Returns true if \p BB is known dead.
  virtual bool isKnownDead(const mlir::Block *BB) const = 0;

  /// Returns true if \p I is assumed dead.
  virtual bool isAssumedDead(const mlir::Operation *I) const = 0;

  /// Returns true if \p I is known dead.
  virtual bool isKnownDead(const mlir::Operation *I) const = 0;

  /// Return true if the underlying value is a store that is known to be
  /// removable. This is different from dead stores as the removable store
  /// can have an effect on live values, especially loads, but that effect
  /// is propagated which allows us to remove the store in turn.
  virtual bool isRemovableStore() const { return false; }

  /// This method is used to check if at least one instruction in a collection
  /// of instructions is live.
  template <typename T> bool isLiveInstSet(T begin, T end) const {
    for (const auto &I : llvm::make_range(begin, end)) {
      //assert(I->getFunction() == getIRPosition().getAssociatedFunction() &&
      //       "Instruction must be in the same anchor scope function.");

      if (!isAssumedDead(I))
        return true;
    }

    return false;
  }

public:
  /// Create an abstract attribute view for the position \p IRP.
  static AAIsDead &createForPosition(const IRPosition &IRP, Attributor &A);

  /// Determine if \p F might catch asynchronous exceptions.
//  static bool mayCatchAsynchronousExceptions(const Function &F) {
//    return F.hasPersonalityFn() && !canSimplifyInvokeNoUnwind(&F);
//  }

  /// Return if the edge from \p From BB to \p To BB is assumed dead.
  /// This is specifically useful in AAReachability.
  virtual bool isEdgeDead(const mlir::Block *From,
                          const mlir::Block *To) const {
    return false;
  }

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAIsDead"; }

  /// See AbstractAttribute::getIdAddr()
  const char *getIdAddr() const override { return &ID; }

  /// This function should return true if the type of the \p AA is AAIsDead
  static bool classof(const AbstractElement *AA) {
    return (AA->getIdAddr() == &ID);
  }

  /// Unique ID (due to the unique address)
  static const char ID;

  friend class Attributor;
};

} // namespace rust_compiler::analysis::attributer
