#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

namespace rust_compiler::crate_builder {

mlir::Value CrateBuilder::emitExpressionWithoutBlock(
    std::shared_ptr<ast::Expression> expr) {
  auto withOut = std::static_pointer_cast<ast::ExpressionWithoutBlock>(expr);

  switch (withOut->getWithoutBlockKind()) {
  case ast::ExpressionWithoutBlockKind::LiteralExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::PathExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::OperatorExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::GroupedExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::ArrayExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::AwaitExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::IndexExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::TupleExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::TupleIndexingExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::StructExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::CallExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::MethodCallExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::FieldExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::ClosureExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::AsyncBlockExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::ContinueExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::BreakExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::RangeExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::ReturnExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::UnderScoreExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::MacroInvocation: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
