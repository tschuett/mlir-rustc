#include "Analysis/MemorySSA.h"

#include "MemorySSANode.h"

#include "llvm/ADT/STLExtras.h"

#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace rust_compiler::analysis {

static auto hasMemEffect(mlir::Operation &op) {
  struct Result {
    bool read = false;
    bool write = false;
  };

  Result ret;
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Write>())
      ret.write = true;

    if (effects.hasEffect<mlir::MemoryEffects::Read>())
      ret.read = true;
  } else if (op.hasTrait<mlir::OpTrait::HasRecursiveSideEffects>()) {
    ret.write = true;
  }
  return ret;
}

Node *memSSAProcessRegion(mlir::Region &region, Node *entryNode,
                          MemorySSA &memSSA) {
  assert(nullptr != entryNode);
  // Only structured control flow is supported for now
  if (!llvm::hasSingleElement(region))
    return nullptr;

  auto &block = region.front();
  Node *currentNode = entryNode;
  for (auto &op : block) {
    if (!op.getRegions().empty()) {
      if (auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op)) {
      } else if (auto branchReg =
                     mlir::dyn_cast<mlir::RegionBranchOpInterface>(op)) {
      }
    } else {
      auto res = hasMemEffect(op);
      if (res.write) {
        auto newNode = memSSA.createDef(&op, currentNode);
        newNode->setDominator(currentNode);
        currentNode->setPostDominator(newNode);
        currentNode = newNode;
      }
      if (res.read)
        memSSA.createUse(&op, currentNode);
    }
  }

  return nullptr;
}

std::optional<MemorySSA> buildMemorySSA(::mlir::Region &region) {
  MemorySSA ret;
  if (auto last = memSSAProcessRegion(region, ret.getRoot(), ret)) {
    ret.getTerm()->setArgument(0, last);
  } else {
    return std::nullopt;
  }
  return std::move(ret);
}

} // namespace rust_compiler::analysis

// https://discourse.llvm.org/t/upstreaming-from-our-mlir-python-compiler-project/64931/4
