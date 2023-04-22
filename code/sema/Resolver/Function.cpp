#include "AST/FunctionParameters.h"
#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <set>

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveFunction(std::shared_ptr<ast::Function> fun,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {

  CanonicalPath segment =
      CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  tyCtx->insertCanonicalPath(fun->getNodeId(), cpath);

  resolveVisibility(fun->getVisibility());

  NodeId scopeNodeId = fun->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getTypeScope().peek());

  if (fun->hasGenericParams())
    resolveGenericParams(fun->getGenericParams(), prefix, canonicalPrefix);

  if (fun->hasWhereClause())
    resolveWhereClause(fun->getWhereClause());

  if (fun->hasReturnType())
    resolveType(fun->getReturnType());

  FunctionParameters params = fun->getParams();
  assert(!params.hasSelfParam() && "to be implemented");

  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<NodeId>())};

  for (auto &parm : params.getParams()) {
    switch (parm.getKind()) {
    case FunctionParamKind::Pattern: {
      FunctionParamPattern pattern = parm.getPattern();
      if (pattern.hasType()) {
        resolveType(pattern.getType());
        resolvePatternDeclarationWithBindings(pattern.getPattern(),
                                              RibKind::Parameter, bindings);
      } else {
        assert(false && "to be implemented");
      }
      break;
    }
    case FunctionParamKind::DotDotDot: {
      assert(false && "to be implemented");
    }
    case FunctionParamKind::Type: {
      assert(false && "to be implemented");
    }
    }
  }

  llvm::errs() << "resolve block epxression"
               << "\n";

  resolveExpression(fun->getBody(), prefix, canonicalPrefix);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

} // namespace rust_compiler::sema::resolver
