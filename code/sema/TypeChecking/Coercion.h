#pragma once

#include "Autoderef.h"
#include "Location.h"
#include "TyTy.h"

#include <vector>

namespace rust_compiler::sema::type_checking {

class CoercionResult {
  TyTy::BaseType *type;

  std::vector<Adjustment> adjustments;

public:
  TyTy::BaseType *getType() const { return type; }

  std::vector<Adjustment> getAdjustments() const { return adjustments; }

  bool isError() const {
    return type == nullptr || type->getKind() == TyTy::TypeKind::Error;
  }
};

class Coercion {
public:
  CoercionResult coercion(TyTy::BaseType *receiver, TyTy::BaseType *expected,
                          Location loc, bool allowAutoderef);

private:
};

TyTy::BaseType *coercion(basic::NodeId, TyTy::WithLocation lhs,
                         TyTy::WithLocation rhs, Location unify);

TyTy::BaseType *coercionWithSite(basic::NodeId, TyTy::WithLocation lhs,
                                 TyTy::WithLocation rhs, Location unify);

} // namespace rust_compiler::sema::type_checking
