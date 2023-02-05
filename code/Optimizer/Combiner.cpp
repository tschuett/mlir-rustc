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

class CmpIOpPattern : public RewritePattern {
public:
  CmpIOpPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(::mlir::arith::CmpIOp::getOperationName(), _benefit,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

class AndIOpPattern : public RewritePattern {
public:
  AndIOpPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(::mlir::arith::AndIOp::getOperationName(), _benefit,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

class OrIOpPattern : public RewritePattern {
public:
  OrIOpPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(::mlir::arith::AndIOp::getOperationName(), _benefit,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

class XorIOpPattern : public RewritePattern {
public:
  XorIOpPattern(mlir::PatternBenefit _benefit, MLIRContext *context)
      : RewritePattern(::mlir::arith::XOrIOp::getOperationName(), _benefit,
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

LogicalResult CmpIOpPattern ::matchAndRewrite(Operation *op,
                                              PatternRewriter &rewriter) const {
  if (auto cmpi = mlir::dyn_cast<::mlir::arith::CmpIOp>(op)) {
    mlir::arith::CmpIPredicate pred = cmpi.getPredicate();
    if (auto conLOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
            cmpi.getLhs().getDefiningOp())) {
      if (auto conROp = mlir::dyn_cast<mlir::arith::ConstantOp>(
              cmpi.getRhs().getDefiningOp())) {
        if (auto srcRAttr = conROp.getValue().cast<IntegerAttr>()) {
          if (auto srcLAttr = conLOp.getValue().cast<IntegerAttr>()) {

            llvm::APInt rhs = srcRAttr.getValue();
            llvm::APInt lhs = srcLAttr.getValue();

            mlir::arith::ConstantOp result;
            switch (pred) {
            case mlir::arith::CmpIPredicate::eq: {
              if (lhs == rhs)
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::ne: {
              if (lhs != rhs)
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::slt: {
              if (lhs.slt(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::sgt: {
              if (lhs.sgt(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::sge: {
              if (lhs.sge(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::sle: {
              if (lhs.sle(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::ult: {
              if (lhs.ult(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::ule: {
              if (lhs.ule(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::ugt: {
              if (lhs.ugt(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            case mlir::arith::CmpIPredicate::uge: {
              if (lhs.uge(rhs))
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
              else
                result = rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(rewriter.getI1Type(), 0));
              op->replaceAllUsesWith(result);
              return success();
            }
            }
          }
        }
      }
    }
  }
  return failure();
}

LogicalResult AndIOpPattern::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  if (auto andi = mlir::dyn_cast<::mlir::arith::AndIOp>(op)) {
    if (auto conLOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
            andi.getLhs().getDefiningOp())) {
      if (auto conROp = mlir::dyn_cast<mlir::arith::ConstantOp>(
              andi.getRhs().getDefiningOp())) {
        if (auto srcRAttr = conROp.getValue().cast<IntegerAttr>()) {
          if (auto srcLAttr = conLOp.getValue().cast<IntegerAttr>()) {
            llvm::APInt result = srcLAttr.getValue();
            result &= srcRAttr.getValue();
            mlir::arith::ConstantOp op2 =
                rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(andi.getLhs().getType(), result));
            op->replaceAllUsesWith(op2);
            return success();
          }
        }
      }
    }
  }
  return failure();
}

LogicalResult OrIOpPattern::matchAndRewrite(Operation *op,
                                            PatternRewriter &rewriter) const {
  if (auto ori = mlir::dyn_cast<::mlir::arith::OrIOp>(op)) {
    if (auto conLOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
            ori.getLhs().getDefiningOp())) {
      if (auto conROp = mlir::dyn_cast<mlir::arith::ConstantOp>(
              ori.getRhs().getDefiningOp())) {
        if (auto srcRAttr = conROp.getValue().cast<IntegerAttr>()) {
          if (auto srcLAttr = conLOp.getValue().cast<IntegerAttr>()) {
            llvm::APInt result = srcLAttr.getValue();
            result |= srcRAttr.getValue();
            mlir::arith::ConstantOp op2 =
                rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(ori.getLhs().getType(), result));
            op->replaceAllUsesWith(op2);
            return success();
          }
        }
      }
    }
  }
  return failure();
}

LogicalResult XorIOpPattern::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  if (auto xori = mlir::dyn_cast<::mlir::arith::XOrIOp>(op)) {
    if (auto conLOp = mlir::dyn_cast<mlir::arith::ConstantOp>(
            xori.getLhs().getDefiningOp())) {
      if (auto conROp = mlir::dyn_cast<mlir::arith::ConstantOp>(
              xori.getRhs().getDefiningOp())) {
        if (auto srcRAttr = conROp.getValue().cast<IntegerAttr>()) {
          if (auto srcLAttr = conLOp.getValue().cast<IntegerAttr>()) {
            llvm::APInt result = srcLAttr.getValue();
            result ^= srcRAttr.getValue();
            mlir::arith::ConstantOp op2 =
                rewriter.create<mlir::arith::ConstantOp>(
                    op->getLoc(),
                    rewriter.getIntegerAttr(xori.getLhs().getType(), result));
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
  rewritePatterns.add<CmpIOpPattern>(PatternBenefit(1), context);
  rewritePatterns.add<AndIOpPattern>(PatternBenefit(1), context);
  rewritePatterns.add<OrIOpPattern>(PatternBenefit(1), context);
  rewritePatterns.add<XorIOpPattern>(PatternBenefit(1), context);

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
