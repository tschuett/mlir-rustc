#include "AST/LoopExpression.h"

#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeLoopExpression(std::shared_ptr<ast::LoopExpression> arith) {
  switch (arith->getLoopExpressionKind()) {
  case ast::LoopExpressionKind::InfiniteLoopExpression: {
    analyzeInfiniteLoopExpression(std::static_pointer_cast<InfiniteLoopExpression>(arith));
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
