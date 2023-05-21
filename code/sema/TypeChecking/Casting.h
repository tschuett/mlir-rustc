#pragma once

#include "TyCtx/TyTy.h"

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::tyctx;

class Casting {
public:
  static TyTy::BaseType *castWithSite(basic::NodeId id, TyTy::WithLocation lhs,
                                      TyTy::WithLocation rhs, Location loc);

private:
  TyTy::BaseType *cast(basic::NodeId id, TyTy::WithLocation lhs,
                       TyTy::WithLocation rhs, Location loc);
};

} // namespace rust_compiler::sema::typechecking
