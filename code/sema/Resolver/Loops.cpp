#include "ADT/CanonicalPath.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/LoopExpression.h"
#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveLoopExpression(
    std::shared_ptr<ast::LoopExpression> loop, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (loop->getLoopExpressionKind()) {
  case LoopExpressionKind::InfiniteLoopExpression: {
    resolveInfiniteLoopExpression(
        std::static_pointer_cast<InfiniteLoopExpression>(loop), prefix,
        canonicalPrefix);
    break;
  }
  case LoopExpressionKind::PredicateLoopExpression: {
    assert(false && "to be handled later");
  }
  case LoopExpressionKind::PredicatePatternLoopExpression: {
    assert(false && "to be handled later");
  }
  case LoopExpressionKind::IteratorLoopExpression: {
    assert(false && "to be handled later");
  }
  case LoopExpressionKind::LabelBlockExpression: {
    assert(false && "to be handled later");
  }
  }
}

void Resolver::resolveInfiniteLoopExpression(
    std::shared_ptr<ast::InfiniteLoopExpression> infini,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (infini->hasLabel()) {
    LoopLabel l = infini->getLabel();
    Identifier name = l.getName();
    NodeId id = l.getNodeId();
    getLabelScope().insert(CanonicalPath::newSegment(infini->getNodeId(), name),
                           id, l.getLocation(), RibKind::Label);
  }

  resolveBlockExpression(
      std::static_pointer_cast<BlockExpression>(infini->getBody()), prefix,
      canonicalPrefix);
}

} // namespace rust_compiler::sema::resolver
