#include "Analysis/Attributer/Attributer.h"

#include "Analysis/Attributer/AANoop.h"
#include "Analysis/Attributer/Common.h"
#include "Analysis/Attributer/DependencyGraphNode.h"
#include "mlir/Support/LogicalResult.h"

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

using namespace llvm;
using namespace mlir;

namespace rust_compiler::analysis::attributor {

void Attributor::setup() {
  module.walk([&](mlir::func::FuncOp op) {
//    getOrCreateAAFor<AANoop>(IRPosition::forFuncOp(op), nullptr,
//                             DepClass::NONE);
  });

  //    if (mlir::func::FuncOp fun = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
  //          } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
  //
  // }
  //    {
  //      IRPosition FPos = IRPosition::forFuncOp(fun);
  //      getOrCreateAAFor<AAIsDead>(FPos);
  //      // for(auto& rs : fun.getResultTypes()) {
  //      //
  //      // }
  //    }
}

mlir::LogicalResult Attributor::runTillFixpoint() {
  llvm::SmallVector<AbstractElement *, 32> changedElements;
  // SmallVector<AbstractElement *, 32> invalidElements;
  llvm::SetVector<AbstractElement *> worklist;

  // FIXME documentation
  /// fix point iteration
  do {

    //// invalid elements
    // for (size_t i = 0; i < invalidElements.size(); ++i) {
    //   auto *invalidElement = invalidElements[i];
    //
    //   while (not invalidElement->deps.empty()) {
    //     const auto &dep = invalidElement->deps.back();
    //     invalidElement->deps.pop_back();
    //     auto *dependentElement = cast<AbstractElement>(dep.getPointer());
    //     if (dep.getInt() == static_cast<unsigned>(DepClass::OPTIONAL)) {
    //       worklist.insert(dependentElement);
    //       continue;
    //     }
    //     dependentElement->getState().indicatePessimisticFixpoint();
    //     assert(dependentElement->getState().isAtFixpoint() &&
    //            "expected fixpoint state");
    //     if (!dependentElement->getState().isValidState()) {
    //       invalidElements.push_back(dependentElement);
    //     } else {
    //       changedElements.push_back(dependentElement);
    //     }
    //   }
    // }

    // Add all abstract elements that are potentially dependent on one that
    // changed to the work list.
    for (auto *changedElement : changedElements) {
      while (!depGraph.areDepsEmpty(changedElement)) {
        // while (!changedElement->deps.empty()) {
        if (DependencyGraphNode *node =
                depGraph.pickDependency(changedElement)) {
          worklist.insert(cast<AbstractElement>(node));
          depGraph.removeDependency(changedElement, node);
        }
      }
    }

    // Reset the changed set.
    changedElements.clear();

    // Update all abstract elements in the work list and record the ones that
    // changed.
    for (auto *element : worklist) {
      const auto &elementState = element->getState();
      if (!elementState.isAtFixpoint()) {
        if (updateElement(*element) == ChangeStatus::CHANGED) {
          changedElements.push_back(element);
        }
      }
    }

    // Add elements to the changed set if they have been created in the last
    // iteration.

    llvm::SetVector<DependencyGraphNode *> newNodes = depGraph.getNewNodes();
    depGraph.resetNewNodes();

    //FIXME: changedElements.append(newNodes.begin(), newNodes.end());

    // Reset the work list and repopulate with the changed abstract elements.
    // Note that dependent ones have already been added above.
    worklist.clear();
    worklist.insert(changedElements.begin(), changedElements.end());

  } while (!worklist.empty());

  return LogicalResult::success();
}

ChangeStatus Attributor::updateElement(AbstractElement &element) {
  assert(phase == Phase::UPDATE &&
         "can update element only in the update stage");

  // Perform the abstract element update.
  [[maybe_unused]]auto &elementState = element.getState();
  ChangeStatus changeStatus = element.update(*this);

  return changeStatus;
}

void Attributor::recordDependence(const AbstractElement &FromAA,
                                  const AbstractElement &ToAA,
                                  DepClass DepClass) {
  assert(false);
  if (DepClass == DepClass::NONE)
    return;

  if (FromAA.getState().isAtFixpoint())
    return;

  depGraph.addDependency(&FromAA, &ToAA, DepClass);
}

// Attributor::Attributor(mlir::ModuleOp module) {}

const IRPosition IRPosition::EmptyKey(Kind::Block,
                                      llvm::DenseMapInfo<void *>::getEmptyKey(),
                                      0);

const IRPosition
    IRPosition::TombstoneKey(Kind::Block,
                             llvm::DenseMapInfo<void *>::getTombstoneKey(), 0);

} // namespace rust_compiler::analysis::attributor
