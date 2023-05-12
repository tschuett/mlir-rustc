#include "AST/ArrayElements.h"
#include "AST/ArrayExpression.h"
#include "AST/Expression.h"
#include "AST/GroupedExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

/// https://doc.rust-lang.org/reference/expressions.html#place-expressions-and-value-expressions

bool Sema::isPlaceExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock:
    return isPlaceExpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
  case ExpressionKind::ExpressionWithoutBlock:
    return isPlaceExpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
  }
}

bool Sema::isValueExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock:
    return isValueExpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
  case ExpressionKind::ExpressionWithoutBlock:
    return isValueExpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
  }
}

bool Sema::isPlaceExpressionWithBlock(ast::ExpressionWithBlock *) {
  assert(false);
}

bool Sema::isPlaceExpressionWithoutBlock(ast::ExpressionWithoutBlock *woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false);
  }
  }
}

bool Sema::isValueExpressionWithBlock(ast::ExpressionWithBlock *) {
  assert(false);
}

bool Sema::isValueExpressionWithoutBlock(ast::ExpressionWithoutBlock *woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false);
  }
  }
}

bool Sema::isAssigneeExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock:
    return isAssigneeExpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
  case ExpressionKind::ExpressionWithoutBlock:
    return isAssigneeExpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
  }
}

bool Sema::isAssigneeExpressionWithBlock(ast::ExpressionWithBlock *) {
  return false;
}

bool Sema::isAssigneeExpressionWithoutBlock(
    ast::ExpressionWithoutBlock *woBlock) {
  switch (woBlock->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false);
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false);
  }
  }
}

} // namespace rust_compiler::sema
