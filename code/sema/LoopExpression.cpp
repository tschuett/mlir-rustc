#include "AST/LoopExpression.h"

#include "AST/PredicatePatternLoopExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeLoopExpression(ast::LoopExpression *arith) {
  switch (arith->getLoopExpressionKind()) {
  case ast::LoopExpressionKind::InfiniteLoopExpression: {
    analyzeInfiniteLoopExpression(static_cast<InfiniteLoopExpression *>(arith));
    break;
  }
  case ast::LoopExpressionKind::PredicateLoopExpression: {
    break;
  }
  case ast::LoopExpressionKind::PredicatePatternLoopExpression: {
    analyzePredicatePatternLoopExpression(
        static_cast<PredicatePatternLoopExpression *>(arith));
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

void Sema::analyzePredicatePatternLoopExpression(
    ast::PredicatePatternLoopExpression *pred) {
  pred->getScrutinee().setPlaceExpression();
}

} // namespace rust_compiler::sema
