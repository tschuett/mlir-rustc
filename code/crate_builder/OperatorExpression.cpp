#include "AST/OperatorExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ComparisonExpression.h"
#include "CrateBuilder/CrateBuilder.h"
#include "TyCtx/TyTy.h"

#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;
using namespace mlir::arith;

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitOperatorExpression(ast::OperatorExpression *expr) {
  switch (expr->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    break;
  }
  case OperatorExpressionKind::DereferenceExpression: {
    break;
  }
  case OperatorExpressionKind::ErrorPropagationExpression: {
    break;
  }
  case OperatorExpressionKind::NegationExpression: {
    break;
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    return emitArithmeticOrLogicalExpression(
        static_cast<ArithmeticOrLogicalExpression *>(expr));
  }
  case OperatorExpressionKind::ComparisonExpression: {
    return emitComparisonExpression(static_cast<ComparisonExpression *>(expr));
    break;
  }
  case OperatorExpressionKind::LazyBooleanExpression: {
    break;
  }
  case OperatorExpressionKind::TypeCastExpression: {
    break;
  }
  case OperatorExpressionKind::AssignmentExpression: {
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    break;
  }
  }
  assert(false);
}

mlir::Value CrateBuilder::emitArithmeticOrLogicalExpression(
    ArithmeticOrLogicalExpression *expr) {
  // using TypeKind = rust_compiler::tyctx::TyTy::TypeKind;

  std::optional<mlir::Value> lhs = emitExpression(expr->getLHS().get());
  std::optional<mlir::Value> rhs = emitExpression(expr->getRHS().get());

  if (!lhs.has_value()) {
    llvm::errs() << "emitExpression in emitArithmeticOrLogicalExpression failed"
                 << "\n";
    exit(1);
  }
  if (!rhs.has_value()) {
    llvm::errs() << "emitExpression in emitArithmeticOrLogicalExpression failed"
                 << "\n";
    exit(1);
  }
  std::optional<tyctx::TyTy::BaseType *> maybeType =
      tyCtx->lookupType(expr->getNodeId());
  bool isIntegerType = false;
  bool isFloatType = false;
  if (maybeType) {
    if (isIntegerLike((*maybeType)->getKind()))
      isIntegerType = true;
    if (isFloatLike((*maybeType)->getKind()))
      isFloatType = true;
  }

  if (!isIntegerType && !isFloatType) {
    llvm::errs() << "expression is neither integer- nor float-like"
                 << "\n";
    exit(1);
  }

  switch (expr->getKind()) {
  case ArithmeticOrLogicalExpressionKind::Addition: {
    if (isFloatType) {
      return builder.create<mlir::arith::AddFOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    } else if (isIntegerType) {
      return builder.create<mlir::arith::AddIOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Subtraction: {
    if (isFloatType) {
      return builder.create<mlir::arith::SubFOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    } else if (isIntegerType) {
      return builder.create<mlir::arith::SubIOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Multiplication: {
    if (isFloatType) {
      return builder.create<mlir::arith::MulFOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    } else if (isIntegerType) {
      return builder.create<mlir::arith::MulFOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Division: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind()))
        return builder.create<mlir::arith::DivSIOp>(
            getLocation(expr->getLocation()), *lhs, *rhs);
      else
        return builder.create<mlir::arith::DivUIOp>(
            getLocation(expr->getLocation()), *lhs, *rhs);

    } else {
      return builder.create<mlir::arith::DivFOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind()))
        return builder.create<mlir::arith::RemSIOp>(
            getLocation(expr->getLocation()), *lhs, *rhs);
      else
        return builder.create<mlir::arith::RemUIOp>(
            getLocation(expr->getLocation()), *lhs, *rhs);

    } else {
      return builder.create<mlir::arith::RemFOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    if (isIntegerType) {
      return builder.create<mlir::arith::AndIOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    if (isIntegerType) {
      return builder.create<mlir::arith::OrIOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    if (isIntegerType) {
      return builder.create<mlir::arith::XOrIOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift: {
    if (isIntegerType) {
      return builder.create<mlir::arith::ShLIOp>(
          getLocation(expr->getLocation()), *lhs, *rhs);
    }
    assert(false);
    break;
  }
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind())) {
        return builder.create<mlir::arith::ShRSIOp>(
            getLocation(expr->getLocation()), *lhs, *rhs);
      } else {
        return builder.create<mlir::arith::ShRUIOp>(
            getLocation(expr->getLocation()), *lhs, *rhs);
      }
    }
    assert(false);
    break;
  }
  }
  assert(false);
}

mlir::Value
CrateBuilder::emitComparisonExpression(ast::ComparisonExpression *expr) {
  // using TypeKind = rust_compiler::tyctx::TyTy::TypeKind;

  std::optional<mlir::Value> lhs = emitExpression(expr->getLHS().get());
  std::optional<mlir::Value> rhs = emitExpression(expr->getRHS().get());

  if (!lhs.has_value()) {
    llvm::errs() << "emitExpression in emitComparisonExpression failed"
                 << "\n";
    exit(1);
  }
  if (!rhs.has_value()) {
    llvm::errs() << "emitExpression in emitComparisonExpression failed"
                 << "\n";
    exit(1);
  }

  bool isIntegerType = false;
  bool isFloatType = false;
  std::optional<tyctx::TyTy::BaseType *> maybeType =
      tyCtx->lookupType(expr->getNodeId());
  if (maybeType) {
    if (isIntegerLike((*maybeType)->getKind()))
      isIntegerType = true;
    if (isFloatLike((*maybeType)->getKind()))
      isFloatType = true;
  }

  if (!isIntegerType && !isFloatType) {
    llvm::errs() << "expression is neither integer- nor float-like"
                 << "\n";
    exit(1);
  }

  switch (expr->getKind()) {
  case ComparisonExpressionKind::Equal: {
    if (isIntegerType) {
      return builder.create<mlir::arith::CmpIOp>(
          getLocation(expr->getLocation()), CmpIPredicate::eq, *lhs, *rhs);
    } else {
      return builder.create<mlir::arith::CmpFOp>(
          getLocation(expr->getLocation()), CmpFPredicate::OEQ, *lhs, *rhs);
    }
    assert(false);
  }
  case ComparisonExpressionKind::NotEqual: {
    if (isIntegerType) {
      return builder.create<mlir::arith::CmpIOp>(
          getLocation(expr->getLocation()), CmpIPredicate::ne, *lhs, *rhs);
    } else {
      return builder.create<mlir::arith::CmpFOp>(
          getLocation(expr->getLocation()), CmpFPredicate::ONE, *lhs, *rhs);
    }
    assert(false);
  }
  case ComparisonExpressionKind::GreaterThan: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind()))
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::sgt, *lhs, *rhs);
      else
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::ugt, *lhs, *rhs);

    } else {
      return builder.create<mlir::arith::CmpFOp>(
          getLocation(expr->getLocation()), CmpFPredicate::OGT, *lhs, *rhs);
    }
    assert(false);
  }
  case ComparisonExpressionKind::LessThan: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind()))
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::slt, *lhs, *rhs);
      else
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::ult, *lhs, *rhs);

    } else {
      return builder.create<mlir::arith::CmpFOp>(
          getLocation(expr->getLocation()), CmpFPredicate::OLT, *lhs, *rhs);
    }
    assert(false);
  }
  case ComparisonExpressionKind::GreaterThanOrEqualTo: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind()))
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::sge, *lhs, *rhs);
      else
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::uge, *lhs, *rhs);

    } else {
      return builder.create<mlir::arith::CmpFOp>(
          getLocation(expr->getLocation()), CmpFPredicate::OGE, *lhs, *rhs);
    }
    assert(false);
  }
  case ComparisonExpressionKind::LessThanOrEqualTo: {
    if (isIntegerType) {
      if (isSignedIntegerLike((*maybeType)->getKind()))
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::sle, *lhs, *rhs);
      else
        return builder.create<mlir::arith::CmpIOp>(
            getLocation(expr->getLocation()), CmpIPredicate::ule, *lhs, *rhs);

    } else {
      return builder.create<mlir::arith::CmpFOp>(
          getLocation(expr->getLocation()), CmpFPredicate::OLE, *lhs, *rhs);
    }
    assert(false);
  }
  }
}

} // namespace rust_compiler::crate_builder
