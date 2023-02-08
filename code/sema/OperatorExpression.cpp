#include "AST/ArithmeticOrLogicalExpression.h"
#include "Sema/TypeChecking.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void TypeChecking::checkArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith) {

  switch(arith->getKind()) {
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

} // namespace rust_compiler::sema


// https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators
