#include "Optimizer/Passes.h"

#include "Hir/HirInterfaces.h"
#include "Hir/HirOps.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;
using namespace rust_compiler::optimizer;

#include "Combine.cpp.inc"

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_COMBINERPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class CondBranchToBranchPattern : public RewritePattern {
public:
  CondBranchToBranchPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(rust_compiler::hir::MutBorrowOp::getOperationName(),
                       _benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

class CombinerPass
    : public rust_compiler::optimizer::impl::CombinerPassBase<CombinerPass> {
public:
  CombinerPass() = default;

  CombinerPass(const CombinerPassOptions &options)
      : CombinerPassBase(options), options(options) {}

  llvm::StringRef getDescription() const override;

  LogicalResult initialize(MLIRContext *context) override;

  void runOnOperation() override;

private:
  FrozenRewritePatternSet frozenPatterns;
  CombinerPassOptions options;
};

} // namespace

using namespace rust_compiler::hir;

LogicalResult
CondBranchToBranchPattern::matchAndRewrite(Operation *op,
                                           PatternRewriter &rewriter) const {
  llvm::outs() << "tryCondBranchToBranchPattern"
               << "\n";
//  if (CondBranchOp cond =
//          mlir::dyn_cast<rust_compiler::Hir::CondBranchOp>(op)) {
//  }
  return failure();
}

llvm::StringRef CombinerPass::getDescription() const { return "combiner pass"; }

LogicalResult CombinerPass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);

  populateWithGenerated(rewritePatterns);

  rewritePatterns.add<CondBranchToBranchPattern>(PatternBenefit(1), context);

  frozenPatterns = FrozenRewritePatternSet(std::move(rewritePatterns));

  return success();
}

void CombinerPass::runOnOperation() {
  mlir::func::FuncOp fun = getOperation();

  llvm::outs() << "run CombinerPass"
               << "\n";

  LogicalResult result = applyPatternsAndFoldGreedily(fun, frozenPatterns);

  if (result.succeeded()) { // [maybe_unused]
  }
}

// https://reviews.llvm.org/D140415
