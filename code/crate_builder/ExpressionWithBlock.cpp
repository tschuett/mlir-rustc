#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitExpressionWithBlock(ast::ExpressionWithBlock* expr) {
  switch (expr->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    //return emitBlockExpression(static_cast<BlockExpression*>(expr));
    assert(false && "to be implemented");
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    return emitLoopExpression(static_cast<LoopExpression*>(expr));
  }
  case ExpressionWithBlockKind::IfExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ExpressionWithBlockKind::IfLetExpression: {
    assert(false && "to be implemented");
    break;
  }
  case ExpressionWithBlockKind::MatchExpression: {
    assert(false && "to be implemented");
    break;
  }
  }
  assert(false);
}

} // namespace rust_compiler::crate_builder
