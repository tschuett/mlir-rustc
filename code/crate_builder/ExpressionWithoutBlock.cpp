#include "AST/AwaitExpression.h"
#include "AST/CallExpression.h"
#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"
#include "Hir/HirOps.h"

#include <memory>

using namespace rust_compiler::ast;

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
    return emitOperatorExpression(
        std::static_pointer_cast<OperatorExpression>(withOut));
    break;
  }
  case ast::ExpressionWithoutBlockKind::GroupedExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::ArrayExpression: {
    break;
  }
  case ast::ExpressionWithoutBlockKind::AwaitExpression: {
    std::shared_ptr<ast::AwaitExpression> await =
        std::static_pointer_cast<ast::AwaitExpression>(withOut);
    mlir::Value body = emitExpression(await->getBody());
    //    return builder.create<hir::AwaitOp>(getLocation(await->getLocation()),
    //    ,body);
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
    std::shared_ptr<ast::CallExpression> call =
        std::static_pointer_cast<ast::CallExpression>(withOut);
    return emitCallExpression(call);
    break;
  }
  case ast::ExpressionWithoutBlockKind::MethodCallExpression: {
    std::shared_ptr<ast::MethodCallExpression> call =
        std::static_pointer_cast<ast::MethodCallExpression>(withOut);
    return emitMethodCallExpression(call);
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

mlir::Value
CrateBuilder::emitCallExpression(std::shared_ptr<ast::CallExpression> expr) {}

mlir::Value CrateBuilder::emitMethodCallExpression(
    std::shared_ptr<ast::MethodCallExpression> expr) {}

} // namespace rust_compiler::crate_builder
