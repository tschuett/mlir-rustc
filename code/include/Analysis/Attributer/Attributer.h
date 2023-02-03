#pragma once

#include "Analysis/Attributer/AbstractElement.h"
#include "Analysis/Attributer/DependencyGraph.h"
#include "Analysis/Attributer/IRPosition.h"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::analysis::attributor {

// Fixed point iteration solver ("monotone framework").
// http://symbolaris.com/course/Compilers11/27-monframework.pdf
class Attributor {
  mlir::ModuleOp module;

  /// A nested map to lookup abstract attributes based on the argument position
  /// on the outer level, and the addresses of the static member (AAType::ID) on
  /// the inner level.
  ///{
  using AAMapKeyTy = std::pair<const char *, IRPosition>;
  llvm::DenseMap<AAMapKeyTy, AbstractElement *> AAMap;
  ///}

  /// A flag that indicates which stage of the process we are in. Initially, the
  /// phase is SEEDING. Phase is changed in `Attributor::run()`
  enum class AttributorPhase {
    SEEDING,
    UPDATE,
    MANIFEST,
    CLEANUP,
  } Phase = AttributorPhase::SEEDING;

  ///}

  /// Return the attribute of \p AAType for \p IRP if existing and valid. This
  /// also allows non-AA users lookup.
  template <typename AAType>
  AAType *lookupAAFor(const IRPosition &IRP,
                      const AbstractElement *QueryingAA = nullptr,
                      DepClass DepClass = DepClass::OPTIONAL,
                      bool AllowInvalidState = false) {
    static_assert(std::is_base_of<AbstractElement, AAType>::value,
                  "Cannot query an attribute with a type not derived from "
                  "'AbstractAttribute'!");
    // Lookup the abstract attribute of type AAType. If found, return it after
    // registering a dependence of QueryingAA on the one returned attribute.
    AbstractElement *AAPtr = AAMap.lookup({&AAType::ID, IRP});
    if (!AAPtr)
      return nullptr;

    AAType *AA = static_cast<AAType *>(AAPtr);

    // Do not register a dependence on an attribute with an invalid state.
    if (DepClass != DepClass::NONE && QueryingAA &&
        AA->getState().isValidState())
      recordDependence(*AA, const_cast<AbstractElement &>(*QueryingAA),
                       DepClass);

    // Return nullptr if this attribute has an invalid state.
    if (!AllowInvalidState && !AA->getState().isValidState())
      return nullptr;
    return AA;
  }

public:
  Attributor(mlir::ModuleOp module) : module(module) {}

  template <typename AAType>
  const AAType &getOrCreateAAFor(IRPosition IRP,
                                 const AbstractElement *QueryingAA,
                                 DepClass DepClass, bool ForceUpdate = false,
                                 bool UpdateAfterInit = true) {
    if (AAType *AAPtr = lookupAAFor<AAType>(IRP, QueryingAA, DepClass,
                                            /* AllowInvalidState */ true)) {
      if (ForceUpdate && Phase == AttributorPhase::UPDATE)
        updateAA(*AAPtr);
      return *AAPtr;
    }

    // No matching attribute found, create one.
    // Use the static create method.
    auto &AA = AAType::createForPosition(IRP, *this);

    // Always register a new attribute to make sure we clean up the allocated
    // memory properly.
    registerAA(AA);

    // If we are currenty seeding attributes, enforce seeding rules.
    if (Phase == AttributorPhase::SEEDING && !shouldSeedAttribute(AA)) {
      AA.getState().indicatePessimisticFixpoint();
      return AA;
    }

    AA.initialize(*this);

    if (QueryingAA && AA.getState().isValidState())
      recordDependence(AA, const_cast<AbstractElement &>(*QueryingAA),
                       DepClass);
    return AA;
  }

  void setup();

  mlir::LogicalResult run() {
    phase = Phase::UPDATE;
    auto result = runTillFixpoint();
    phase = Phase::DONE;

    return result;
  }

private:
  /// Run `::update` on \p AA and track the dependences queried while doing so.
  /// Also adjust the state if we know further updates are not necessary.
  ChangeStatus updateAA(AbstractElement &AA);

  /// This method should be used in conjunction with the `getAAFor` method and
  /// with the DepClass enum passed to the method set to None. This can
  /// be beneficial to avoid false dependences but it requires the users of
  /// `getAAFor` to explicitly record true dependences through this method.
  /// The \p DepClass flag indicates if the dependence is striclty necessary.
  /// That means for required dependences, if \p FromAA changes to an invalid
  /// state, \p ToAA can be moved to a pessimistic fixpoint because it required
  /// information from \p FromAA but none are available anymore.
  void recordDependence(const AbstractElement &FromAA,
                        const AbstractElement &ToAA, DepClass DepClass);

  mlir::LogicalResult runTillFixpoint();

  ChangeStatus updateElement(AbstractElement &element);

  // A flag that indicates which stage of the process we are in.
  enum class Phase {
    // Initial elements are being registered to seed the graph.
    SEEDING,
    // Fixed point iteration is running.
    UPDATE,
    // Iteration has completed; does not indicate whether it coverged.
    DONE,
  } phase = Phase::SEEDING;

  DepdendencyGraph depGraph;
};

} // namespace rust_compiler::analysis::attributor
