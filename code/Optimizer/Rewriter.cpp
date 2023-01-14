#include "Mir/MirDialect.h"
#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_REWRITEPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class EliminateBorrowPattern : public RewritePattern {
public:
  /// This overload constructs a pattern that only matches operations with the
  /// root name of `MyOp`.
  EliminateBorrowPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(rust_compiler::Mir::BorrowOp::getOperationName(),
                       benefit, context) {}

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using the separate hooks.
  LogicalResult match(Operation *op) const override;
  void rewrite(Operation *op, PatternRewriter &rewriter) const override;
};

class EliminateMutBorrowPattern : public RewritePattern {
public:
  EliminateMutBorrowPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(rust_compiler::Mir::MutBorrowOp::getOperationName(),
                       benefit, context) {}

  LogicalResult match(Operation *op) const override;
  void rewrite(Operation *op, PatternRewriter &rewriter) const override;
};

class RewritePass
    : public rust_compiler::optimizer::impl::RewritePassBase<RewritePass> {
public:
  RewritePass() = default;
  RewritePass(const RewritePass &pass);

  llvm::StringRef getDescription() const override;

  LogicalResult initialize(MLIRContext *context) override;

  void runOnOperation() override;

private:
  FrozenRewritePatternSet frozenPatterns;
};

} // namespace

LogicalResult EliminateBorrowPattern::match(Operation *op) const {
  if (mlir::isa<rust_compiler::Mir::BorrowOp>(op))
    return success();
  return failure();
}

void EliminateBorrowPattern::rewrite(Operation *op,
                                     PatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
}

LogicalResult EliminateMutBorrowPattern::match(Operation *op) const {
  if (mlir::isa<rust_compiler::Mir::MutBorrowOp>(op))
    return success();
  return failure();
}

void EliminateMutBorrowPattern::rewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
}

RewritePass::RewritePass(const RewritePass &pass)
    : rust_compiler::optimizer::impl::RewritePassBase<RewritePass>(pass) {}

llvm::StringRef RewritePass::getDescription() const { return "test pass"; }

LogicalResult RewritePass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);

  rewritePatterns.add<EliminateBorrowPattern>(PatternBenefit(1), context);
  rewritePatterns.add<EliminateMutBorrowPattern>(PatternBenefit(1), context);

  frozenPatterns = FrozenRewritePatternSet(std::move(rewritePatterns));

  return success();
}

void RewritePass::runOnOperation() {
  mlir::func::FuncOp fun = getOperation();

  LogicalResult result = applyPatternsAndFoldGreedily(fun, frozenPatterns);

  if (result.succeeded()) {
  }
}

std::unique_ptr<mlir::Pass> createRewritePass() {
  return std::make_unique<RewritePass>();
}

// https://reviews.llvm.org/D140415
