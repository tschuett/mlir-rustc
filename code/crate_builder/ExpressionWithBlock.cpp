#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitExpressionWithBlock(std::shared_ptr<ast::Expression> expr) {

  std::shared_ptr<ast::ExpressionWithBlock> block =
      std::static_pointer_cast<ExpressionWithBlock>(expr);
  switch (block->getWithBlockKind()) {
  case ExpressionWithBlockKind::BlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::UnsafeBlockExpression: {
    break;
  }
  case ExpressionWithBlockKind::LoopExpression: {
    return emitLoopExpression(std::static_pointer_cast<LoopExpression>(block));
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
  assert(false);
}

} // namespace rust_compiler::crate_builder
