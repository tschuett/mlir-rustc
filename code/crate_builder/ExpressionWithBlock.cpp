#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitExpressionWithBlock(ast::ExpressionWithBlock* expr) {

  switch (expr->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    return emitLoopExpression(static_cast<LoopExpression*>(expr));
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
  assert(false);
}

} // namespace rust_compiler::crate_builder
