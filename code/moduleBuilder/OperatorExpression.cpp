#include "AST/ArithmeticOrLogicalExpression.h"
#include "Mir/MirDialect.h"
#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"
#include <mlir/IR/Location.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace rust_compiler {

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

  switch (expr->getKind()) {
  case ast::ArithmeticOrLogicalExpressionKind::Addition: {
    return builder.create<mlir::arith::AddIOp>(location, lhs, rhs);
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
