#include "AST/OperatorExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value CrateBuilder::emitOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> expr) {

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
        std::static_pointer_cast<ArithmeticOrLogicalExpression>(expr));
    break;
  }
  case OperatorExpressionKind::ComparisonExpression: {
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
}

mlir::Value CrateBuilder::emitArithmeticOrLogicalExpression(
    std::shared_ptr<ArithmeticOrLogicalExpression> expr) {
  switch (expr->getKind()) {
  case ArithmeticOrLogicalExpressionKind::Addition: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Subtraction: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Multiplication: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Division: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift: {
    break;
  }
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
