#include "AST/ArrayElements.h"
#include "AST/ArrayExpression.h"
#include "AST/Expression.h"
#include "AST/GroupedExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

bool Sema::isConstantExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ExpressionKind::ExpressionWithBlock:
    return isConstantEpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
  case ExpressionKind::ExpressionWithoutBlock:
    return isConstantEpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
  }
}

bool Sema::isConstantEpressionWithoutBlock(
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
    GroupedExpression *group = static_cast<GroupedExpression*>(woBlock);
    return isConstantExpression(group->getExpression().get());
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    // FIXME no Drop
    ast::ArrayExpression *array = static_cast<ArrayExpression *>(woBlock);
    ArrayElements elements = array->getArrayElements();
    switch (elements.getKind()) {
    case ArrayElementsKind::List: {
      for (auto &expr : elements.getElements())
        if (!isConstantExpression(expr.get()))
          return false;
      return true;
    }
    case ArrayElementsKind::Repeated: {
      return isConstantExpression(elements.getValue().get()) and
             isConstantExpression(elements.getCount().get());
    }
    }
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

bool Sema::isConstantEpressionWithBlock(ast::ExpressionWithBlock *) {
  assert(false);
}

} // namespace rust_compiler::sema
