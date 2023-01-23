#pragma once

#include "Analysis/Attributer/AbstractElement.h"
#include "Analysis/Attributer/DependencyGraph.h"
#include "Analysis/Attributer/IRPosition.h"
#include <mlir/IR/Operation.h>

#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::analysis::attributor {

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

  enum class DepClass {
    REQUIRED, ///< The target cannot be valid if the source is not.
    OPTIONAL, ///< The target may be valid if the source is not.
    NONE,     ///< Do not track a dependence between source and target.
  };
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
  Attributor(mlir::ModuleOp module);

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
  }

  template <typename AAType>
  const AAType &getOrCreateAAFor(const IRPosition &IRP) {
    return getOrCreateAAFor<AAType>(IRP, /* QueryingAA */ nullptr,
                                    DepClass::NONE);
  }

  void setup();

  void run();

private:
  /// Run `::update` on \p AA and track the dependences queried while doing so.
  /// Also adjust the state if we know further updates are not necessary.
  ChangeStatus updateAA(AbstractElement &AA);
};

} // namespace rust_compiler::analysis::attributor
