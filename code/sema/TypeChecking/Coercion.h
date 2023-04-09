#pragma once

#include "Location.h"
#include "Sema/Autoderef.h"
#include "TyCtx/TyTy.h"

#include <vector>

namespace rust_compiler::sema::type_checking {

/// https://doc.rust-lang.org/reference/type-coercions.html

  using namespace rust_compiler::tyctx;
  
class CoercionResult {
  std::vector<Adjustment> adjustments;
  TyTy::BaseType *type = nullptr;

public:
  CoercionResult(std::vector<Adjustment> adjustments, TyTy::BaseType *type)
      : adjustments(std::move(adjustments)), type(type){};
  CoercionResult() = default;

  TyTy::BaseType *getType() const { return type; }

  std::vector<Adjustment> getAdjustments() const { return adjustments; }

  bool isError() const {
    return type == nullptr || type->getKind() == TyTy::TypeKind::Error;
  }
};

class Coercion {
  CoercionResult result;

public:
  CoercionResult coercion(TyTy::BaseType *receiver, TyTy::BaseType *expected,
                          Location loc, bool allowAutoderef);

private:
  bool coerceToNever(TyTy::BaseType *receiver, TyTy::BaseType *expected);
};

TyTy::BaseType *coercion(basic::NodeId, TyTy::WithLocation lhs,
                         TyTy::WithLocation rhs, Location unify);

TyTy::BaseType *coercionWithSite(basic::NodeId, TyTy::WithLocation lhs,
                                 TyTy::WithLocation rhs, Location unify);

} // namespace rust_compiler::sema::type_checking
