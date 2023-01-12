#include "AST/Expression.h"
#include "AST/OperatorExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <memory>
#include <optional>

using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> expr) {

  llvm::outs() << "emitExpressionWithoutBlock"
               << "\n";

  switch (expr->getWithoutBlockKind()) {
  case ast::ExpressionWithoutBlockKind::LiteralExpression: {
    return emitLiteralExpression(
        std::static_pointer_cast<ast::LiteralExpression>(expr));
  }
  case ast::ExpressionWithoutBlockKind::ReturnExpression: {
    emitReturnExpression(std::static_pointer_cast<ast::ReturnExpression>(expr));
    return mlir::Value();
  }
  case ast::ExpressionWithoutBlockKind::OperatorExpression: {
    return emitOperatorExpression(
        std::static_pointer_cast<ast::OperatorExpression>(expr));
  }
  default: {
    assert(false);
  }
  }

  // FIXME
  return nullptr;
}

} // namespace rust_compiler
