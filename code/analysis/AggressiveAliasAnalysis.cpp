#include "Analysis/AggressiveAliasAnalysis.h"

#include <cstdlib>
#include <llvm/ADT/StringRef.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace mlir;

namespace rust_compiler::analysis {

// AliasSet AliasSet::getSubset(mlir::Value address) const {
//   std::set<std::pair<MemoryAccess, MemoryAccess>> subset;
//
//   for (auto &par : mayAliases) {
//     if (par.first.mayAlias(address))
//       subset.insert(par);
//     else if (par.second.mayAlias(address))
//       subset.insert(par);
//   }
//
//   return AliasSet(subset);
// }

AggressiveAliasAnalysis::AggressiveAliasAnalysis(mlir::ModuleOp &mod) {
  initialize(mod);
}

mlir::AliasResult AggressiveAliasAnalysis::alias(mlir::Value lhs,
                                                 mlir::Value rhs) {
  //  if (lhs == rhs)
  // FIXME
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
      changed |= analyzeFunction(f);
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

bool AggressiveAliasAnalysis::analyzeFunction(mlir::func::FuncOp *f) {
  Function *fun = functions[f->getName()].get();
  // iterate
  bool changed = false;
  do {
    for (Block &block : f->getBody()) {
      for (Operation &op : block) {
        if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
          if (op.getNumResults() != 1) {
            llvm::errs() << "load with more than one result"
                         << "\n";
            exit(EXIT_FAILURE);
          }
          load.getResult();
          transferFunLoad(fun, Load(load.getMemRef(), load.getResult()));
        } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
          if (op.getNumResults() != 1) {
            llvm::errs() << "store with more than one result"
                         << "\n";
            exit(EXIT_FAILURE);
          }
          transferFunStore(fun, Store(store.getValue(), store.getMemRef()));
        } else if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
        } else if (auto free = mlir::dyn_cast<mlir::memref::DeallocOp>(op)) {
        } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          //
          // std::string callee = call.getCallee();
        } else if (auto effectInterface =
                       dyn_cast<MemoryEffectOpInterface>(op)) {
          SmallVector<MemoryEffects::EffectInstance, 1> effects;
          effectInterface.getEffects(effects);
          for (const MemoryEffects::EffectInstance &it : effects) {
            if (isa<MemoryEffects::Allocate>(it.getEffect())) {
              it.getValue();
            } else if (isa<MemoryEffects::Free>(it.getEffect())) {
              it.getValue();
            } else if (isa<MemoryEffects::Read>(it.getEffect())) {
              if (op.getNumResults() != 1) {
                llvm::errs() << "load with more than one result"
                             << "\n";
                exit(EXIT_FAILURE);
              }
              it.getValue();
            } else if (isa<MemoryEffects::Write>(it.getEffect())) {
              if (op.getNumResults() != 1) {
                llvm::errs() << "store with more than one result"
                             << "\n";
                exit(EXIT_FAILURE);
              }
              it.getValue();
            }
          }
        }
      }
    }
  } while (changed);
  // hack
  return true;
}

//   if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {

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

  // hack
  return ModRefResult::getModAndRef();
}

void AggressiveAliasAnalysis::transferFunLoad(Function *f, const Load &load) {
  AliasSet Aout;

  AliasSet Ain = f->getEntryAliasSet();
  // AliasSet AinOfP = f->getEntryAliasSet().getSubset(load.getValue());
  /*
    (A_{IN} - A_{IN}(ma)) u
   */
}

void AggressiveAliasAnalysis::transferFunStore(Function *f,
                                               const Store &store) {
  AliasSet Aout;

  AliasSet Ain = f->getEntryAliasSet();
  AliasSet AinOfP = Ain.getSubset(store.getAddress());
  AliasSet AinOfQ = Ain.getSubset(store.getValue());
  AliasSet tmp = Ain.minus(AinOfP);

  AliasSet leftU = Ain.whereLeftIs(store.getAddress());

  /*
    (A_{IN} - (set difference) A_{IN}(ma)) union
   */
  AliasSet result;
  for (AliasRelation &l : leftU.getSets())
    for (AliasRelation &r : AinOfQ.getSets())
      result.add(r.replaceWith(l.getLeft(), store.getValue()));

  f->setExitSet(tmp.join(result));
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

/*
  instead iterate until convergence, iterate with budget
 */
