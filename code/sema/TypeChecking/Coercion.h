#pragma once

#include "Location.h"
#include "TyTy.h"

#include <vector>

namespace rust_compiler::sema::type_checking {

class CommitSite {};
class InferenceSite {};

class Coercon {
public:
  TyTy::BaseType *coercion(TyTy::WithLocation lhs, TyTy::WithLocation rhs,
                           Location loc, bool commit, bool emitErrors,
                           bool inference, std::vector<CommitSite> &commits,
                           std::vector<InferenceSite> &infers);

private:
};

TyTy::BaseType *coercion(basic::NodeId, TyTy::WithLocation lhs,
                        TyTy::WithLocation rhs, Location unify);

} // namespace rust_compiler::sema::type_checking
