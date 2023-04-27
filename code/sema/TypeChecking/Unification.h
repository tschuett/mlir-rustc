#pragma once

#include "Location.h"
#include "TyCtx/TyTy.h"

#include <vector>

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::tyctx;

class CommitSite {};
class InferenceSite {};

class Unification {
public:
  TyTy::BaseType *unify(TyTy::WithLocation lhs, TyTy::WithLocation rhs,
                        Location loc, bool commit, bool emitErrors,
                        bool inference, std::vector<CommitSite> &commits,
                        std::vector<InferenceSite> &infers);

private:
  TyTy::BaseType *expectIntType(TyTy::IntType *left, TyTy::BaseType *right);
  TyTy::BaseType *expectUSizeType(TyTy::USizeType *left, TyTy::BaseType *right);
};

TyTy::BaseType *unify(basic::NodeId, TyTy::WithLocation lhs,
                      TyTy::WithLocation rhs, Location unify);

TyTy::BaseType *unifyWithSite(basic::NodeId, TyTy::WithLocation lhs,
                              TyTy::WithLocation rhs, Location unify);

} // namespace rust_compiler::sema::type_checking
