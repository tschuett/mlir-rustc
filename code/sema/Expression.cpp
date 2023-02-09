#include "AST/Expression.h"

#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeExpressionWithBlock(
    std::shared_ptr<ast::ExpressionWithBlock> let) {
  switch (let->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    analyzeBlockExpression(std::static_pointer_cast<BlockExpression>(let));
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    analyzeLoopExpression(std::static_pointer_cast<LoopExpression>(let));
    break;
  }
  case ExpressionWithBlockKind::IfExpression: {
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    break;
  }
  }
}

void Sema::analyzeExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> let) {
  switch (let->getWithoutBlockKind()) {
  case ExpressionWithoutBlockKind::LiteralExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::PathExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::OperatorExpression: {
    analyzeOperatorExpression(
        std::static_pointer_cast<OperatorExpression>(let));
    break;
  }
  case ExpressionWithoutBlockKind::GroupedExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ArrayExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::AwaitExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::IndexExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::TupleExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::TupleIndexingExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::StructExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::CallExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::MethodCallExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::FieldExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ClosureExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::AsyncBlockExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ContinueExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::BreakExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::RangeExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::ReturnExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::UnderScoreExpression: {
    break;
  }
  case ExpressionWithoutBlockKind::MacroInvocation: {
    break;
  }
  }
}

void Sema::analyzeExpression(std::shared_ptr<ast::Expression> let) {
  switch (let->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    analyzeExpressionWithBlock(
        std::static_pointer_cast<ExpressionWithBlock>(let));
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    analyzeExpressionWithoutBlock(
        std::static_pointer_cast<ExpressionWithoutBlock>(let));
    break;
  }
  }
}

} // namespace rust_compiler::sema
