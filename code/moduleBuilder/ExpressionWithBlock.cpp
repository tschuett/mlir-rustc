#include "AST/BlockExpression.h"
#include "AST/Expression.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <memory>

namespace rust_compiler {

using namespace rust_compiler::ast;

mlir::Value ModuleBuilder::emitExpressionWithBlock(
    std::shared_ptr<ExpressionWithBlock> expr) {
  switch (expr->getKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    std::optional<mlir::Value> result = emitBlockExpression(expr);
    if (result)
      return *result;
    return nullptr;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
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
  // FIXME
  return nullptr;
}

} // namespace rust_compiler
