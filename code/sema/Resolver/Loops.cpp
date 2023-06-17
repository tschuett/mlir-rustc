#include "ADT/CanonicalPath.h"
#include "AST/InfiniteLoopExpression.h"
#include "AST/LoopExpression.h"
#include "Basic/Ids.h"
#include "PatternDeclaration.h"
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
    resolveIteratorLoopExpression(
        std::static_pointer_cast<IteratorLoopExpression>(loop), prefix,
        canonicalPrefix);
    break;
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

void Resolver::resolveIteratorLoopExpression(
    std::shared_ptr<ast::IteratorLoopExpression> iter,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (iter->hasLabel()) {
    LoopLabel l = iter->getLabel();
    Identifier name = l.getName();
    NodeId id = l.getNodeId();

    getLabelScope().insert(CanonicalPath::newSegment(iter->getNodeId(), name),
                           id, l.getLocation(), RibKind::Label);
  }

  basic::NodeId scopeNodeId = iter->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);

  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<NodeId>())};

  for (const auto &p : iter->getPattern()->getPatterns()) {
    PatternDeclaration pat = {p,      RibKind::Variable, bindings, this,
                              prefix, canonicalPrefix};
    pat.resolve();
  }
  //  resolvePatternDeclaration(iter->getPattern(), RibKind::Variable, prefix,
  //                            canonicalPrefix);
  resolveExpression(iter->getRHS(), prefix, canonicalPrefix);
  resolveExpression(iter->getBody(), prefix, canonicalPrefix);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

} // namespace rust_compiler::sema::resolver
