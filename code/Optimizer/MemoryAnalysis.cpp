#include "Analysis/AggressiveAliasAnalysis.h"
#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <cstddef>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <optional>
#include <set>

using namespace mlir;
using namespace rust_compiler::analysis;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_MEMORYANALYSISPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {

class MemoryAnalyzer {

public:
  MemoryAnalyzer(mlir::AliasAnalysis &aliasAnalysis,
                 mlir::DominanceInfo &dominance)
      : aliasAnalysis(aliasAnalysis), dominance(dominance) {}

  void analyze(ModuleOp mod);

private:
  std::optional<mlir::AliasResult> mayAlias(mlir::Operation *a,
                                            mlir::Operation *b);
  mlir::AliasAnalysis &aliasAnalysis;
  mlir::DominanceInfo &dominance;

  bool areArithRelated(mlir::Value, mlir::Value);
  std::optional<size_t> distance(mlir::Value, mlir::Value, int64_t upperBound);
};

class MemoryAnalysisPass
    : public rust_compiler::optimizer::impl::MemoryAnalysisPassBase<
          MemoryAnalysisPass> {
public:
  void runOnOperation() override;
};

} // namespace

/// a and b must be memrefs
std::optional<mlir::AliasResult> MemoryAnalyzer::mayAlias(mlir::Operation *a,
                                                          mlir::Operation *b) {
  mlir::Value valueA;
  mlir::Value valueB;
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(a))
    valueA = load.getMemRef();
  else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(a))
    valueA = store.getMemRef();
  else
    return std::nullopt;

  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(b))
    valueB = load.getMemRef();
  else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(b))
    valueB = store.getMemRef();
  else
    return std::nullopt;

  mlir::AliasResult res = aliasAnalysis.alias(valueA, valueB);

  return res;
}

void MemoryAnalyzer::analyze(ModuleOp mod) {
  for (Block &block : mod.getBodyRegion()) {
    for (Operation &op : block) {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op)) {
        for (Block &block : mod.getBodyRegion()) {
          for (Operation &op2 : block) {
            if (auto maybeDominatedLoad =
                    mlir::dyn_cast<mlir::memref::LoadOp>(op2)) {
              if (dominance.dominates(&op, &op2)) {
                areArithRelated(load.getMemref(),
                                maybeDominatedLoad.getMemref());
              }
            } else if (auto maybeDominatedStore =
                           mlir::dyn_cast<mlir::memref::StoreOp>(op2)) {
              if (dominance.dominates(&op, &op2)) {
                areArithRelated(load.getMemref(),
                                maybeDominatedStore.getMemref());
              }
            }
          }
        }
      }
    }
  }
}

std::optional<size_t> MemoryAnalyzer::distance(mlir::Value from, mlir::Value to,
                                               int64_t upperBound) {
  if (upperBound == 0 or upperBound == -1)
    return std::nullopt;

  if (from == to)
    return 0;
//  for (mlir::Operation *use : from.getUses()) {
//    for (auto &arg : use->getOpOperands()) {
//      if (to == arg.get())
//        return 1;
//      for (auto &arg : use->getOpOperands()) {
//        std::optional<size_t> d = distance(arg.get(), to, upperBound - 1);
//        if (d)
//          return 1 + d;
//      }
//    }
//  }
  return std::nullopt;
}

bool MemoryAnalyzer::areArithRelated(mlir::Value from, mlir::Value to) {
  if (from == to)
    return true;
  return true;
}

void MemoryAnalysisPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  mlir::DataFlowSolver solver;

  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<mlir::dataflow::SparseConstantPropagation>();

  LogicalResult success = solver.initializeAndRun(module);
  if (success.succeeded()) {
    // hide warning
  }

  AggressiveAliasAnalysis aggressive = {module};
  AliasAnalysis &alias = getAnalysis<mlir::AliasAnalysis>();
  alias.addAnalysisImplementation(aggressive);

  MemoryAnalyzer ana = {alias, getAnalysis<mlir::DominanceInfo>()};
  ana.analyze(module);
}

/*
  related!?! memops. Maybe on the same struct?. How does it help?

  two memref ops (a and b): (a dominates b, nope). address not the
  same and addresses are related by arith dialect. The types are equal
  -> do not alias. same struct?
 */
// Handle the case where lhs is a constant.
//Attribute lhsAttr, rhsAttr;
//if (matchPattern(lhs, m_Constant(&lhsAttr))) {
