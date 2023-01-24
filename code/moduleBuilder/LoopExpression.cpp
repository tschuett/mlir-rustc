#include "AST/LoopExpression.h"

#include "AST/PredicateLoopExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitLoopExpression(
    std::shared_ptr<ast::LoopExpression> loopExpr) {

  switch (loopExpr->getLoopExpressionKind()) {
  case LoopExpressionKind::InfiniteLoopExpression: {
    std::shared_ptr<InfiniteLoopExpression> loop =
        std::static_pointer_cast<InfiniteLoopExpression>(loopExpr);

    return emitInfiniteLoopExpression(loop);
  }
  case LoopExpressionKind::PredicateLoopExpression: {
    std::shared_ptr<PredicateLoopExpression> loop =
        std::static_pointer_cast<PredicateLoopExpression>(loopExpr);

    return emitPredicateLoopExpression(loop);
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

  assert(false);
}

} // namespace rust_compiler
