#include "AST/LoopExpression.h"

#include "AST/IteratorLoopExpression.h"
#include "CrateBuilder/CrateBuilder.h"

#include <cassert>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value CrateBuilder::emitLoopExpression(ast::LoopExpression *expr) {
  switch (expr->getLoopExpressionKind()) {
  case LoopExpressionKind::InfiniteLoopExpression: {
    break;
  }
  case LoopExpressionKind::PredicateLoopExpression: {
    break;
  }
  case LoopExpressionKind::PredicatePatternLoopExpression: {
    break;
  }
  case LoopExpressionKind::IteratorLoopExpression: {
    return emitIteratorLoopExpression(
        static_cast<ast::IteratorLoopExpression *>(expr));
  }
  case LoopExpressionKind::LabelBlockExpression: {
    break;
  }
  }
  assert(false);
}

mlir::Value
CrateBuilder::emitIteratorLoopExpression(ast::IteratorLoopExpression *loop) {
  bool isConstant = isConstantExpression(loop->getRHS().get());
  if (isConstant) {
  } else {
    std::optional<mlir::Value> expr = emitExpression(loop->getRHS().get());
    if (expr) {
    }
  }

  assert(false);
}

} // namespace rust_compiler::crate_builder

// FIXME LoopLabel
