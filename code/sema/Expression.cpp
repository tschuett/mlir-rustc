#include "AST/Expression.h"

#include "AST/ArrayElements.h"
#include "AST/IfLetExpression.h"
#include "AST/MatchExpression.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeExpressionWithBlock(ast::ExpressionWithBlock *let) {
  switch (let->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    analyzeBlockExpression(static_cast<BlockExpression *>(let));
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    analyzeLoopExpression(static_cast<LoopExpression *>(let));
    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    analyzeIfExpression(static_cast<IfExpression *>(let));
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    analyzeIfLetExpression(static_cast<IfLetExpression *>(let));
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    analyzeMatchExpression(static_cast<MatchExpression *>(let));
    break;
  }
  }
}

void Sema::analyzeExpressionWithoutBlock(ast::ExpressionWithoutBlock *let) {
  switch (let->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    analyzeOperatorExpression(static_cast<OperatorExpression *>(let));
    break;
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    analyzeArrayExpression(static_cast<ArrayExpression *>(let));
    break;
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false);
    break;
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false);
    break;
  }
  }
}

void Sema::analyzeExpression(ast::Expression *let) {
  switch (let->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    analyzeExpressionWithBlock(static_cast<ExpressionWithBlock *>(let));
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    analyzeExpressionWithoutBlock(static_cast<ExpressionWithoutBlock *>(let));
    break;
  }
  }
}

void Sema::analyzeArrayExpression(ast::ArrayExpression *array) {
  ArrayElements elements = array->getArrayElements();

  switch (elements.getKind()) {
  case ArrayElementsKind::List: {
    break;
  }
  case ArrayElementsKind::Repeated: {
    std::shared_ptr<Expression> count = elements.getCount();

    [[maybe_unused]] bool constant = isConstantExpression(count.get());
    break;
  }
  }
}

void Sema::analyzeMatchExpression(ast::MatchExpression *match) {
  match->getScrutinee().setPlaceExpression();
}

void Sema::analyzeIfLetExpression(ast::IfLetExpression *ifLet) {
  ifLet->getScrutinee().setPlaceExpression();
}

void Sema::analyzeIfExpression(ast::IfExpression *ifExpr) {}

} // namespace rust_compiler::sema
