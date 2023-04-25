#include "AST/ClosureParam.h"
#include "AST/ClosureParameters.h"
#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <set>
#include <vector>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

void Resolver::resolveClosureExpression(
    std::shared_ptr<ast::ClosureExpression> closure,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  assert(false && "to be handled later");

  NodeId scopeNodeId = closure->getNodeId();

  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);

  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<NodeId>())};

  if (closure->hasParameters()) {
    ClosureParameters params = closure->getParameters();
    for (ClosureParam &pa : params.getParameters())
      resolveClosureParameter(pa, bindings, prefix, canonicalPrefix);
  }

  if (closure->hasReturnType())
    resolveType(closure->getReturnType(), prefix, canonicalPrefix);

  pushClosureContext(closure->getNodeId());

  resolveExpression(closure->getBody(), prefix, canonicalPrefix);

  popClosureContext();

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

void Resolver::resolveClosureParameter(
    ClosureParam &param, std::vector<PatternBinding> &bindings,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  resolvePatternDeclarationWithBindings(param.getPattern(), RibKind::Parameter,
                                        bindings, prefix, canonicalPrefix);

  if (param.hasType())
    resolveType(param.getType(), prefix, canonicalPrefix);
}

} // namespace rust_compiler::sema::resolver
