#include "AST/OperatorExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema {

void Sema::analyzeAssignmentExpression(
    std::shared_ptr<AssignmentExpression> arith) {
  NodeId left = getNodeId(arith->getLHS());
  NodeId right = getNodeId(arith->getRHS());
  analyzeExpression(arith->getLHS());
  analyzeExpression(arith->getRHS());

  // FIXME
}

void Sema::analyzeArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith) {

  NodeId left = getNodeId(arith->getLHS());
  NodeId right = getNodeId(arith->getRHS());

  switch (arith->getKind()) {
  case ArithmeticOrLogicalExpressionKind::Addition: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Subtraction: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Multiplication: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Division: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  }
}

void Sema::analyzeOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> arith) {
  switch (arith->getKind()) {
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
    analyzeAssignmentExpression(
        std::static_pointer_cast<AssignmentExpression>(arith));
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    break;
  }
  }
}

} // namespace rust_compiler::sema

// https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators
