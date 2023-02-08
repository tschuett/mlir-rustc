#include "AST/Expression.h"

#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitExpression(std::shared_ptr<ast::Expression> expr) {
  switch (expr->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    return emitExpressionWithBlock(expr);
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    return emitExpressionWithoutBlock(expr);
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
