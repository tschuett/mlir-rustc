#include "Unification.h"

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *Unification::unify(TyTy::WithLocation lhs,
                                   TyTy::WithLocation rhs, Location loc,
                                   bool commit, bool emitErrors, bool inference,
                                   std::vector<CommitSite> &commits,
                                   std::vector<InferenceSite> &infers) {

  assert(false && "to be implemented");
}

TyTy::BaseType *unify(basic::NodeId, TyTy::WithLocation lhs,
                      TyTy::WithLocation rhs, Location unify) {
  assert(false && "to be implemented");

  std::vector<CommitSite> commits;
  std::vector<InferenceSite> infers;

  Unification uni;
  return uni.unify(lhs, rhs, unify, true /*commit*/, true /*emit error*/,
                   false
                   /*infer*/,
                   commits, infers);
}

} // namespace rust_compiler::sema::type_checking
