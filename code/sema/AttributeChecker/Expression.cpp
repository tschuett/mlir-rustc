#include "AST/Expression.h"

#include "AST/BlockExpression.h"
#include "AttributeChecker.h"

using namespace rust_compiler::sema::attribute_checker;

void AttributeChecker::checkExpression(Expression *e) {
  switch (e->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock:
    checkExpressionWithBlock(static_cast<ExpressionWithBlock *>(e));
    break;
  case ExpressionKind::ExpressionWithoutBlock:
    checkExpressionWithoutBlock(static_cast<ExpressionWithoutBlock *>(e));
    break;
  }
}

void AttributeChecker::checkExpressionWithBlock(
    ExpressionWithBlock *withBlock) {
  switch (withBlock->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression:
    checkBlockExpression(static_cast<BlockExpression *>(withBlock));
    break;
  case ExpressionWithBlockKind::UnsafeBlockExpression:
    break;
  case ExpressionWithBlockKind::LoopExpression:
    break;
  case ExpressionWithBlockKind::IfExpression:
    break;
  case ExpressionWithBlockKind::IfLetExpression:
    break;
  case ExpressionWithBlockKind::MatchExpression:
    break;
  }
}

void AttributeChecker::checkExpressionWithoutBlock(
    ExpressionWithoutBlock *woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression:
    break;
  case ExpressionWithoutBlockKind::PathExpression:
    break;
  case ExpressionWithoutBlockKind::OperatorExpression:
    break;
  case ExpressionWithoutBlockKind::GroupedExpression:
    break;
  case ExpressionWithoutBlockKind::ArrayExpression:
    break;
  case ExpressionWithoutBlockKind::AwaitExpression:
    break;
  case ExpressionWithoutBlockKind::IndexExpression:
    break;
  case ExpressionWithoutBlockKind::TupleExpression:
    break;
  case ExpressionWithoutBlockKind::TupleIndexingExpression:
    break;
  case ExpressionWithoutBlockKind::StructExpression:
    break;
  case ExpressionWithoutBlockKind::CallExpression:
    break;
  case ExpressionWithoutBlockKind::MethodCallExpression:
    break;
  case ExpressionWithoutBlockKind::FieldExpression:
    break;
  case ExpressionWithoutBlockKind::ClosureExpression:
    break;
  case ExpressionWithoutBlockKind::AsyncBlockExpression:
    break;
  case ExpressionWithoutBlockKind::ContinueExpression:
    break;
  case ExpressionWithoutBlockKind::BreakExpression:
    break;
  case ExpressionWithoutBlockKind::RangeExpression:
    break;
  case ExpressionWithoutBlockKind::ReturnExpression:
    break;
  case ExpressionWithoutBlockKind::UnderScoreExpression:
    break;
  case ExpressionWithoutBlockKind::MacroInvocation:
    break;
  }
}

void AttributeChecker::checkBlockExpression(BlockExpression *block) {}
