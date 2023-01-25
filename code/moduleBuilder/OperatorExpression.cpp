#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/AssignmentExpression.h"
#include "Mir/MirDialect.h"
#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include "mlir/IR/TypeRange.h"

#include <mlir/IR/Location.h>

using namespace rust_compiler::Mir;
using namespace llvm;
using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> opr) {

  switch (opr->getKind()) {
  case OperatorExpressionKind::BorrowExpression: {
    return emitBorrowExpression(static_pointer_cast<BorrowExpression>(opr));
  }
  case OperatorExpressionKind::ComparisonExpression: {
    return emitComparisonExpression(
        static_pointer_cast<ComparisonExpression>(opr));
  }
  case OperatorExpressionKind::NegationExpression: {
    return emitNegationExpression(static_pointer_cast<NegationExpression>(opr));
  }
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    return emitArithmeticOrLogicalExpression(
        static_pointer_cast<ArithmeticOrLogicalExpression>(opr));
  }
  case OperatorExpressionKind::AssignmentExpression: {
    return emitAssignmentExpression(
        static_pointer_cast<AssignmentExpression>(opr));
  }
  default: {
    assert(false);
  }
  }

  return nullptr;
}

mlir::Value ModuleBuilder::emitBorrowExpression(
    std::shared_ptr<ast::BorrowExpression> borrow) {
  mlir::Value rhs = emitExpression(borrow->getExpression());
  if (borrow->isMutable()) {
    return builder.create<Mir::MutBorrowOp>(
        getLocation(borrow->getLocation()),
        mlir::TypeRange(builder.getI64Type()), rhs);
  } else {
    return builder.create<Mir::BorrowOp>(getLocation(borrow->getLocation()),
                                         mlir::TypeRange(builder.getI64Type()),
                                         rhs);
  }
  // FIXME: types
}

mlir::Value ModuleBuilder::emitAssignmentExpression(
    std::shared_ptr<ast::AssignmentExpression>) {
  assert(false);
}

mlir::Value ModuleBuilder::emitNegationExpression(
    std::shared_ptr<ast::NegationExpression>) {
  assert(false);
}

mlir::Value ModuleBuilder::emitArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> expr) {
  mlir::Value lhs = emitExpression(expr->getLHS());

  if (!lhs)
    return nullptr;

  mlir::Value rhs = emitExpression(expr->getRHS());

  if (!rhs)
    return nullptr;

  Location loc = expr->getLHS()->getLocation();

  mlir::Location location = getLocation(loc);

  // FIXME
  // auto type = getType(expr->getType());
  mlir::Type type = builder.getIntegerType(64, false);

  switch (expr->getKind()) {
  case ast::ArithmeticOrLogicalExpressionKind::Addition: {
    return builder.create<rust_compiler::Mir::AddIOp>(location, type, lhs, rhs);
  }
  case ast::ArithmeticOrLogicalExpressionKind::Subtraction: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::Multiplication: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::Division: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::Remainder: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::LeftShift: {
    break;
  }
  case ast::ArithmeticOrLogicalExpressionKind::RightShift: {
    break;
  }
  }

  // FIXME
  return nullptr;
}

} // namespace rust_compiler
