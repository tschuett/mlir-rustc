#include "Resolver.h"
#include "Mappings/Mappings.h"
#include "llvm/Support/raw_ostream.h"

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::basic;
using namespace rust_compiler::mappings;

namespace rust_compiler::sema::resolver {

void Resolver::resolveFunction(std::shared_ptr<ast::Function> fun,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {

  llvm::errs() << "name resolution in function" << "\n";
  
  CanonicalPath segment =
      CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
  CanonicalPath path = prefix.append(segment);
  CanonicalPath cpath = canonicalPrefix.append(segment);

  Mappings::get()->insertCanonicalPath(fun->getNodeId(), cpath);

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

  for (auto &param : fun->getParams()) {
    resolveType(param.getType());
    resolvePatternDeclaration(param.getPattern(), RibKind::Parameter);
  }

  resolveExpression(fun->getBody(), prefix, canonicalPrefix);

  getNameScope().pop();
  getTypeScope().pop();
  getLabelScope().pop();
}

} // namespace rust_compiler::sema::resolver
