#include "AST/PathIdentSegment.h"
#include "Basic/Ids.h"
#include "PathProbing.h"
#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

std::optional<TyTy::BaseType *>
TypeResolver::resolveOperatorOverload(PropertyKind kind, Expression *expr,
                                      TyTy::BaseType *lhs,
                                      TyTy::BaseType *rhs) {

  std::string associatedPropertyName = Property2String(kind);

  //bool context = haveFunctionContext();

  ast::PathIdentSegment ident = ast::PathIdentSegment(expr->getLocation());
  ident.setIdentifier(Identifier(associatedPropertyName));

  std::set<MethodCandidate> candidates = probeMethodResolver(
      lhs, ident, false /*autoderef flag*/);

  assert(false);
}

} // namespace rust_compiler::sema::type_checking
