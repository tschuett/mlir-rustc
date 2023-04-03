#include "Coercion.h"

#include "TyCtx/TyCtx.h"
#include "TyTy.h"
#include "Unification.h"

using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

CoercionResult Coercion::coercion(TyTy::BaseType *receiver,
                                  TyTy::BaseType *expected, Location loc,
                                  bool allowAutoderef) {
  //  llvm::errs() << receiver->toString() << "\n";
  //  llvm::errs() << expected->toString() << "\n";

  bool success = false;
  if (receiver->getKind() == TyTy::TypeKind::Never)
    success = coerceToNever(receiver, expected);
  else {
    assert(false && "to be implemented");
  }
  assert(success);

  return result;
}

bool Coercion::coerceToNever(TyTy::BaseType *receiver,
                             TyTy::BaseType *expected) {
  if (expected->getKind() == TyTy::TypeKind::Int) {
    result = CoercionResult({}, expected);
    return true;
  }
  assert(false && "to be implemented");
}

TyTy::BaseType *coercion(basic::NodeId, TyTy::WithLocation lhs,
                         TyTy::WithLocation rhs, Location unify) {
  assert(false && "to be implemented");
}

TyTy::BaseType *coercionWithSite(basic::NodeId id, TyTy::WithLocation lhs,
                                 TyTy::WithLocation rhs, Location unify) {
  TyTy::BaseType *expectedType = lhs.getType();
  TyTy::BaseType *expression = rhs.getType();

  if (expectedType->getKind() == TyTy::TypeKind::Error ||
      expression->getKind() == TyTy::TypeKind::Error)
    return expression;

  Coercion coercion;
  CoercionResult result = coercion.coercion(expression, expectedType, unify,
                                            true /*allow autoderef*/);

  TyTy::BaseType *receiver = expression;
  if (!result.isError())
    receiver = result.getType();

  TyTy::BaseType *coerced = unifyWithSite(
      id, lhs, TyTy::WithLocation(receiver, rhs.getLocation()), unify);

  TyCtx::get()->insertAutoderefMapping(id, result.getAdjustments());

  return coerced;
}

} // namespace rust_compiler::sema::type_checking
