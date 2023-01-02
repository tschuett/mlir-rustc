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
    return emitBlockExpression(static_pointer_cast<BlockExpression>(expr));
    break;
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
}

} // namespace rust_compiler
