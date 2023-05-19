#include "Analysis/AggressiveAliasAnalysis.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace rust_compiler::analysis;

/// based on TestAliasAnalysis.cpp from mlir

namespace {
struct TestAliasAnalysisPass
    : public PassWrapper<TestAliasAnalysisPass, OperationPass<ModuleOp>> {

  void printAliasOperand(Operation *op) {
    llvm::errs() << op->getAttrOfType<StringAttr>("test.ptr").getValue();
  }
  
  void printAliasOperand(Value value) {
    if (BlockArgument arg = dyn_cast<BlockArgument>(value)) {
      Region *region = arg.getParentRegion();
      unsigned parentBlockNumber =
          std::distance(region->begin(), arg.getOwner()->getIterator());
      llvm::errs() << region->getParentOp()
                          ->getAttrOfType<StringAttr>("test.ptr")
                          .getValue()
                   << ".region" << region->getRegionNumber();
      if (parentBlockNumber != 0)
        llvm::errs() << ".block" << parentBlockNumber;
      llvm::errs() << "#" << arg.getArgNumber();
      return;
    }
    OpResult result = cast<OpResult>(value);
    printAliasOperand(result.getOwner());
    llvm::errs() << "#" << result.getResultNumber();
  }

  void printAliasResult(AliasResult result, Value lhs, Value rhs) {
    printAliasOperand(lhs);
    llvm::errs() << " <-> ";
    printAliasOperand(rhs);
    llvm::errs() << ": " << result << "\n";
  }

  /// Print the result of an alias query.
  void printModRefResult(ModRefResult result, Operation *op, Value location) {
    printAliasOperand(op);
    llvm::errs() << " -> ";
    printAliasOperand(location);
    llvm::errs() << ": " << result << "\n";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    AggressiveAliasAnalysis aggressive = {module};
    aliasAnalysis.addAnalysisImplementation(aggressive);

    llvm::errs() << "Testing : " << module->getAttr("sym_name") << "\n";

    // Collect all of the values to check for aliasing behavior.
    SmallVector<Value, 32> valsToCheck;
    module.walk([&](Operation *op) {
      if (!op->getAttr("test.ptr"))
        return;
      valsToCheck.append(op->result_begin(), op->result_end());
      for (Region &region : op->getRegions())
        for (Block &block : region)
          valsToCheck.append(block.args_begin(), block.args_end());
    });

    // Check for aliasing behavior between each of the values.
    for (auto it = valsToCheck.begin(), e = valsToCheck.end(); it != e; ++it)
      for (auto *innerIt = valsToCheck.begin(); innerIt != it; ++innerIt)
        printAliasResult(aliasAnalysis.alias(*innerIt, *it), *innerIt, *it);

    // Check for aliasing behavior between each of the values.
    for (auto &it : valsToCheck) {
      module.walk([&](Operation *op) {
        if (!op->getAttr("test.ptr"))
          return;
        printModRefResult(aliasAnalysis.getModRef(op, it), op, it);
      });
    }
  };
};

} // namespace

namespace mlir::test {
void registerTestAliasAnalysisPass() {
  PassRegistration<TestAliasAnalysisPass>();
}

} // namespace mlir::test
