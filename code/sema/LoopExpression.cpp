#include "AST/LoopExpression.h"

#include "Sema/Sema.h"

namespace rust_compiler::sema {

void Sema::analyzeLoopExpression(std::shared_ptr<ast::LoopExpression> arith) {
  switch (arith->getLoopExpressionKind()) {
  case ast::LoopExpressionKind::InfiniteLoopExpression: {
    break;
  }
  case ast::LoopExpressionKind::PredicateLoopExpression: {
    break;
  }
  case ast::LoopExpressionKind::PredicatePatternLoopExpression: {
    break;
  }
  case ast::LoopExpressionKind::IteratorLoopExpression: {
    break;
  }
  case ast::LoopExpressionKind::LabelBlockExpression: {
    break;
  }
  }
}

} // namespace rust_compiler::sema
