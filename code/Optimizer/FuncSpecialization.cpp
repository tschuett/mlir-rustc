#include "Optimizer/Passes.h"

#include <cstddef>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/FoldUtils.h>

using namespace mlir;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_FUNCSPECIALPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {

class Constant {
  Type type;
  Attribute attr;
  Dialect *dialect;
  Location loc;

public:
  mlir::Value getValue(OperationFolder &folder, mlir::Block *block) {
    return folder.getOrCreateConstant(block, dialect, attr, type, loc);
  }
};

class FuncSpecialPass
    : public rust_compiler::optimizer::impl::FuncSpecialPassBase<
          FuncSpecialPass> {
public:
  void runOnOperation() override;

private:
  void checkCallOps(mlir::func::FuncOp *f, mlir::OperationFolder *folder);
  void analyzeCallOp(mlir::func::CallOp *c, mlir::OperationFolder *folder);

  MLIRContext *context;
  llvm::DenseSet<mlir::func::FuncOp> internalFuncs;
};

} // namespace

void FuncSpecialPass::analyzeCallOp(mlir::func::CallOp *call,
                                    mlir::OperationFolder *folder) {
  bool inPlaceUpdate;
  /// check args
  for (auto arg : call->getArgOperands()) {
    mlir::Operation *op = arg.getDefiningOp();
    if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      llvm::SmallVector<OpFoldResult, 1> foldedOp;
      LogicalResult result = op->fold(/*operands*/ std::nullopt, foldedOp);
      if (succeeded(result)) {
        [[maybe_unused]]Attribute attr = foldedOp.front().get<Attribute>();
        op->getDialect();
      }
    }
    if (mlir::succeeded(folder->tryToFold(op, &inPlaceUpdate))) {
      if (!inPlaceUpdate) {
        /// xxx
      }
    }
  }
}

void FuncSpecialPass::checkCallOps(mlir::func::FuncOp *f,
                                   mlir::OperationFolder *folder) {
  for (auto &block : f->getBody()) {
    for (auto &op : block.getOperations()) {
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        analyzeCallOp(&call, folder);
      }
    }
  }
}

void FuncSpecialPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  context = module.getContext();
  mlir::OperationFolder folder(context);
  mlir::OpBuilder builder(context);

  module.walk([&](mlir::func::FuncOp f) {
    if (!f.isDeclaration())
      internalFuncs.insert(f);
  });

  module.walk([&](mlir::func::FuncOp f) { checkCallOps(&f, &folder); });
}

/// https://reviews.llvm.org/D78397

/// https://reviews.llvm.org/D93838

// There is funcOp.clone

// https://reviews.llvm.org/D76020: ConstantLike trait
