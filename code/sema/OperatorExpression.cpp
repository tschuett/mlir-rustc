#include "AST/OperatorExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeAssignmentExpression(
    std::shared_ptr<AssignmentExpression> arith) {
  NodeId left = getNodeId(arith->getLHS());
  NodeId right = getNodeId(arith->getRHS());
  typeChecking.eqType(left, right);
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
    typeChecking.eqType(left, right);
    typeChecking.isIntegerOrFloatLike(left);
    typeChecking.isIntegerOrFloatLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Subtraction: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerOrFloatLike(left);
    typeChecking.isIntegerOrFloatLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Multiplication: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerOrFloatLike(left);
    typeChecking.isIntegerOrFloatLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Division: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerOrFloatLike(left);
    typeChecking.isIntegerOrFloatLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerOrFloatLike(left);
    typeChecking.isIntegerOrFloatLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerLike(left);
    typeChecking.isIntegerLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerLike(left);
    typeChecking.isIntegerLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerLike(left);
    typeChecking.isIntegerLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerLike(left);
    typeChecking.isIntegerLike(right);
    analyzeExpression(arith->getLHS());
    analyzeExpression(arith->getRHS());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    typeChecking.eqType(left, right);
    typeChecking.isIntegerLike(left);
    typeChecking.isIntegerLike(right);
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
