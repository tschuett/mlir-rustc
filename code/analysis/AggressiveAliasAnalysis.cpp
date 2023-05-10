#include "Analysis/AggressiveAliasAnalysis.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <llvm/ADT/StringRef.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

using namespace mlir;

namespace rust_compiler::analysis {

AliasSet AliasSet::getSubset(mlir::Value *address) const {
  std::set<std::pair<mlir::Value *, mlir::Value *>> subset;

  for (auto &par : mayAliases) {
    if (par.first == address)
      subset.insert(par);
    else if (par.second == address)
      subset.insert(par);
  }

  return AliasSet(subset);
}

AggressiveAliasAnalysis::AggressiveAliasAnalysis(mlir::ModuleOp &mod) {
  initialize(mod);
}

mlir::AliasResult AggressiveAliasAnalysis::alias(mlir::Value lhs,
                                                 mlir::Value rhs) {
  if (lhs == rhs)
    return AliasResult::MustAlias;
}

void AggressiveAliasAnalysis::initialize(mlir::ModuleOp &mod) {
  // build the initial control flow graph
  buildInitialCfg(mod);
  // initalize the alias sets
  mod.walk([this](mlir::func::FuncOp f) {
    if (f.isDeclaration())
      return;
    functions.insert({f.getName(), std::make_unique<Function>()});
  });

  // iterate
  bool changed = false;
  do {
    for (mlir::func::FuncOp *f : cfg.getFunctions()) {
    }
  } while (changed);
}

void AggressiveAliasAnalysis::buildInitialCfg(mlir::ModuleOp &mod) {
  mod.walk([this](mlir::func::FuncOp f) {
    for (Block &block : f.getBody()) {
      for (Operation &op : block) {
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          cfg.addEdge(f.getName(), callOp.getCallee(), &f);
        }
      }
    }
  });
}

void AggressiveAliasAnalysis::analyzeFunction(mlir::func::FuncOp *f) {
  Function *fun = functions[f->getName()].get();
  // iterate
  bool changed = false;
  do {
    for (Block &block : f->getBody()) {
      for (Operation &op : block) {
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
          load.getMemRef();
        } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
          store.getMemRef();
        } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        }
      }
    }
  } while (changed);
}

/// Return the modify-reference behavior of `op` on `location`.
ModRefResult AggressiveAliasAnalysis::getModRef(Operation *op, Value location) {
  // based on LocalAliasAnalysis

  // Check to see if this operation relies on nested side effects.
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    // TODO: To check recursive operations we need to check all of the nested
    // operations, which can result in a quadratic number of queries. We
    // should introduce some caching of some kind to help alleviate this,
    // especially as this caching could be used in other areas of the codebase
    // (e.g. when checking `wouldOpBeTriviallyDead`).
    return ModRefResult::getModAndRef();
  }
}

} // namespace rust_compiler::analysis

/*

  mlir::memory::AllocOp
  mlir::memory::RellocOp
  mlir::memory::DeallocOp
 */

/*
   pointer assignment == load or store
   <z, y> access path z and y may-aliases
   probably address z and y may-alias -> two mlir::Value s may alias, i.e., two
   addresses
 */
