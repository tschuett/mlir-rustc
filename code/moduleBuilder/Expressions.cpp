#include "AST/Expression.h"
#include "ModuleBuilder/ModuleBuilder.h"

using namespace rust_compiler::ast;

namespace rust_compiler {

void ModuleBuilder::buildExpression(std::shared_ptr<Expression> expr) {
  ExpressionKind kind = expr->getExpressionKind();

  switch (kind) {
  case ExpressionKind::ExpressionWithBlock: {
    buildExpressionWithBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithBlock>(expr));
    break;
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    buildExpressionWithoutBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithoutBlock>(expr));
    break;
  }
  }
}

void ModuleBuilder::buildExpressionWithBlock(
    std::shared_ptr<ast::ExpressionWithBlock> expr) {
  // FIXME
}

} // namespace rust_compiler
