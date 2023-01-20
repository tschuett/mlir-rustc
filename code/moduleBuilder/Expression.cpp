#include "AST/Expression.h"

#include "ModuleBuilder/ModuleBuilder.h"

using namespace llvm;
using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitExpression(std::shared_ptr<Expression> expr) {
  ExpressionKind kind = expr->getExpressionKind();

  switch (kind) {
  case ExpressionKind::ExpressionWithBlock: {
    return emitExpressionWithBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithBlock>(expr));
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    return emitExpressionWithoutBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithoutBlock>(expr));
  }
  }
}

} // namespace rust_compiler
