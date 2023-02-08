#include "AST/LoopExpression.h"

#include "CrateBuilder/CrateBuilder.h"

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitLoopExpression(std::shared_ptr<ast::LoopExpression> expr) {
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
    break;
  }
  case LoopExpressionKind::LabelBlockExpression: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder


// FIXME LoopLabel
