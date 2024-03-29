#include "AST/FunctionParameters.h"
#include "PatternDeclaration.h"
#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <set>

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

void Resolver::resolveFunction(ast::Function *fun,
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
    resolveGenericParams(fun->getGenericParams(), path, cpath);

  if (fun->hasWhereClause())
    resolveWhereClause(fun->getWhereClause(), path, cpath);

  if (fun->hasReturnType())
    resolveType(fun->getReturnType(), path, cpath);

  FunctionParameters params = fun->getParams();
  // assert(!params.hasSelfParam() && "to be implemented");

  std::vector<PatternBinding> bindings = {
      PatternBinding(PatternBoundCtx::Product, std::set<NodeId>())};

  if (fun->hasParams()) {
    for (auto &parm : params.getParams()) {
      switch (parm.getKind()) {
      case FunctionParamKind::Pattern: {
        FunctionParamPattern pattern = parm.getPattern();
        if (pattern.hasType()) {
          resolveType(pattern.getType(), path, cpath);
          PatternDeclaration pat = {pattern.getPattern(),
                                    RibKind::Parameter,
                                    bindings,
                                    this,
                                    path,
                                    cpath};
          pat.resolve();
          //          resolvePatternDeclarationWithBindings(pattern.getPattern(),
          //                                                RibKind::Parameter,
          //                                                bindings, prefix,
          //                                                canonicalPrefix);
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
  }

  if (fun->hasBody())
    resolveExpression(fun->getBody(), path, cpath);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

} // namespace rust_compiler::sema::resolver
