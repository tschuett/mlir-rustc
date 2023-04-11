#include "AST/AwaitExpression.h"
#include "AST/CallExpression.h"
#include "AST/Expression.h"
#include "AST/PathExpression.h"
#include "AST/ReturnExpression.h"
#include "CrateBuilder/CrateBuilder.h"
#include "Hir/HirOps.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

  std::optional<mlir::Value>
CrateBuilder::emitExpressionWithoutBlock(ast::ExpressionWithoutBlock *expr) {
  switch (expr->getWithoutBlockKind()) {
  case ast::ExpressionWithoutBlockKind::LiteralExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::PathExpression: {
    return emitPathExpression(static_cast<PathExpression*>(expr));
  }
  case ast::ExpressionWithoutBlockKind::OperatorExpression: {
    return emitOperatorExpression(static_cast<OperatorExpression *>(expr));
  }
  case ast::ExpressionWithoutBlockKind::GroupedExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ArrayExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::AwaitExpression: {
    //    std::shared_ptr<ast::AwaitExpression> await =
    //        std::static_pointer_cast<ast::AwaitExpression>(withOut);
    //    mlir::Value body = emitExpression(await->getBody());
    //    return builder.create<hir::AwaitOp>(getLocation(await->getLocation()),
    //                                        body);
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::IndexExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::TupleExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::TupleIndexingExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::StructExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::CallExpression: {
    ast::CallExpression *call = static_cast<ast::CallExpression *>(expr);
    return emitCallExpression(call);
  }
  case ast::ExpressionWithoutBlockKind::MethodCallExpression: {
    ast::MethodCallExpression *call =
        static_cast<ast::MethodCallExpression *>(expr);
    return emitMethodCallExpression(call);
  }
  case ast::ExpressionWithoutBlockKind::FieldExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ClosureExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::AsyncBlockExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ContinueExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::BreakExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::RangeExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::ReturnExpression: {
    emitReturnExpression(static_cast<ReturnExpression *>(expr));
    return std::nullopt;
  }
  case ast::ExpressionWithoutBlockKind::UnderScoreExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ast::ExpressionWithoutBlockKind::MacroInvocation: {
    assert(false && "to be implemented");
    break;
  }
  }
  assert(false);
}

// mlir::Value
// CrateBuilder::emitCallExpression(std::shared_ptr<ast::CallExpression> expr)
// {}
//
// mlir::Value CrateBuilder::emitMethodCallExpression(
//     std::shared_ptr<ast::MethodCallExpression> expr) {}

} // namespace rust_compiler::crate_builder
