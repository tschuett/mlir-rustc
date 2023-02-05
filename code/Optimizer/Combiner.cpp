#include "Hir/HirInterfaces.h"
#include "Hir/HirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
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
class AddiOpPattern : public RewritePattern {
public:
  AddiOpPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(::mlir::arith::AddIOp::getOperationName(), _benefit,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

class SubiOpPattern : public RewritePattern {
public:
  SubiOpPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(::mlir::arith::SubIOp::getOperationName(), _benefit,
                       context) {}

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

LogicalResult AddiOpPattern::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  if (auto addi = mlir::dyn_cast<::mlir::arith::AddIOp>(op)) {
    if (auto conLOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
            addi.getLhs().getDefiningOp())) {
      if (auto conROp = mlir::dyn_cast<mlir::arith::ConstantOp>(
              addi.getRhs().getDefiningOp())) {
        if (auto srcRAttr = conROp.getValue().cast<IntegerAttr>()) {
          if (auto srcLAttr = conLOp.getValue().cast<IntegerAttr>()) {
            llvm::APInt result = srcRAttr.getValue() + srcLAttr.getValue();
            mlir::arith::ConstantOp op2 =
                rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(addi.getLhs().getType(), result));
            op->replaceAllUsesWith(op2);
            return success();
          }
        }
      }
    }
  }
  return failure();
}

LogicalResult SubiOpPattern::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  if (auto subi = mlir::dyn_cast<::mlir::arith::SubIOp>(op)) {
    if (auto conLOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
            subi.getLhs().getDefiningOp())) {
      if (auto conROp = mlir::dyn_cast<mlir::arith::ConstantOp>(
              subi.getRhs().getDefiningOp())) {
        if (auto srcRAttr = conROp.getValue().cast<IntegerAttr>()) {
          if (auto srcLAttr = conLOp.getValue().cast<IntegerAttr>()) {
            llvm::APInt result = srcLAttr.getValue() - srcRAttr.getValue();
            mlir::arith::ConstantOp op2 =
                rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(subi.getLhs().getType(), result));
            op->replaceAllUsesWith(op2);
            return success();
          }
        }
      }
    }
  }
  return failure();
}

llvm::StringRef CombinerPass::getDescription() const { return "combiner pass"; }

LogicalResult CombinerPass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);

  populateWithGenerated(rewritePatterns);

  rewritePatterns.add<AddiOpPattern>(PatternBenefit(1), context);
  rewritePatterns.add<SubiOpPattern>(PatternBenefit(1), context);

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
