#include "AST/OperatorExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/AssignmentExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema {

void Sema::analyzeCompoundAssignmentExpression(
    ast::CompoundAssignmentExpression *compound) {

  compound->getLHS()->setPlaceExpression();
}

void Sema::analyzeAssignmentExpression(AssignmentExpression *arith) {
  //  NodeId left = getNodeId(arith->getLHS());
  //  NodeId right = getNodeId(arith->getRHS());
  analyzeExpression(arith->getLHS().get());
  analyzeExpression(arith->getRHS().get());

  // AssignmentExpression *ass = static_cast<AssignmentExpression *>(arith);

  // FIXME
}

void Sema::analyzeArithmeticOrLogicalExpression(
    std::shared_ptr<ast::ArithmeticOrLogicalExpression> arith) {

  //  NodeId left = getNodeId(arith->getLHS());
  //  NodeId right = getNodeId(arith->getRHS());

  switch (arith->getKind()) {
  case ArithmeticOrLogicalExpressionKind::Addition: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Subtraction: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Multiplication: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Division: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::Remainder: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseAnd: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseOr: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::BitwiseXor: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::LeftShift: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  case ArithmeticOrLogicalExpressionKind::RightShift: {
    analyzeExpression(arith->getLHS().get());
    analyzeExpression(arith->getRHS().get());
    break;
  }
  }
}

void Sema::analyzeOperatorExpression(ast::OperatorExpression *arith) {
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
    analyzeAssignmentExpression(static_cast<AssignmentExpression *>(arith));
    break;
  }
  case OperatorExpressionKind::CompoundAssignmentExpression: {
    analyzeCompoundAssignmentExpression(
        static_cast<CompoundAssignmentExpression *>(arith));
    break;
  }
  }
}

} // namespace rust_compiler::sema

// https://doc.rust-lang.org/reference/expressions/operator-expr.html#arithmetic-and-logical-binary-operators
