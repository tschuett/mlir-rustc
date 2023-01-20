#include "AST/ArithmeticOrLogicalExpression.h"
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
  case OperatorExpressionKind::ArithmeticOrLogicalExpression: {
    return emitArithmeticOrLogicalExpression(
        static_pointer_cast<ArithmeticOrLogicalExpression>(opr));
  }
  default: {
    assert(false);
  }
  }

  return nullptr;
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
