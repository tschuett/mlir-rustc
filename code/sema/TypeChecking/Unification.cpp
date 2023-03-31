#include "Unification.h"

#include "TyTy.h"

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *Unification::unify(TyTy::WithLocation lhs,
                                   TyTy::WithLocation rhs, Location loc,
                                   bool commit, bool emitErrors, bool inference,
                                   std::vector<CommitSite> &commits,
                                   std::vector<InferenceSite> &infers) {

  TyTy::BaseType *leftType = lhs.getType();
  TyTy::BaseType *rightType = rhs.getType();

  assert(leftType->getNumberOfSpecifiedBounds() == 0);

  assert(inference == false);

  switch (leftType->getKind()) {
  case TyTy::TypeKind::Bool: {
    assert(false);
  }
  case TyTy::TypeKind::Char: {
    assert(false);
  }
  case TyTy::TypeKind::Int: {
    return expectIntType(static_cast<TyTy::IntType *>(leftType), rightType);
  }
  case TyTy::TypeKind::Uint: {
    assert(false);
  }
  case TyTy::TypeKind::USize: {
    assert(false);
  }
  case TyTy::TypeKind::ISize: {
    assert(false);
  }
  case TyTy::TypeKind::Float: {
    assert(false);
  }
  case TyTy::TypeKind::Closure: {
    assert(false);
  }
  case TyTy::TypeKind::Function: {
    assert(false);
  }
  case TyTy::TypeKind::Inferred: {
    assert(false);
  }
  case TyTy::TypeKind::Never: {
    assert(false);
  }
  case TyTy::TypeKind::Str: {
    assert(false);
  }
  case TyTy::TypeKind::Tuple: {
    assert(false);
  }
  case TyTy::TypeKind::Parameter: {
    assert(false);
  }
  case TyTy::TypeKind::Error: {
    assert(false);
  }
  }
}

TyTy::BaseType *Unification::expectIntType(TyTy::IntType *left,
                                           TyTy::BaseType *right) {
  switch (right->getKind()) {
  case TyTy::TypeKind::Bool: {
    assert(false);
  }
  case TyTy::TypeKind::Char: {
    assert(false);
  }
  case TyTy::TypeKind::Int: {
    TyTy::IntType *rightInt = static_cast<TyTy::IntType *>(right);
    if (rightInt->getIntKind() == left->getIntKind())
      return new TyTy::IntType(left->getTypeReference(), left->getIntKind());
    assert(false);
  }
  case TyTy::TypeKind::Uint: {
    assert(false);
  }
  case TyTy::TypeKind::USize: {
    assert(false);
  }
  case TyTy::TypeKind::ISize: {
    assert(false);
  }
  case TyTy::TypeKind::Float: {
    assert(false);
  }
  case TyTy::TypeKind::Closure: {
    assert(false);
  }
  case TyTy::TypeKind::Function: {
    assert(false);
  }
  case TyTy::TypeKind::Inferred: {
    assert(false);
  }
  case TyTy::TypeKind::Never: {
    assert(false);
  }
  case TyTy::TypeKind::Str: {
    assert(false);
  }
  case TyTy::TypeKind::Tuple: {
    assert(false);
  }
  case TyTy::TypeKind::Parameter: {
    assert(false);
  }
  case TyTy::TypeKind::Error: {
    assert(false);
  }
  }
  assert(false);
}

TyTy::BaseType *unify(basic::NodeId, TyTy::WithLocation lhs,
                      TyTy::WithLocation rhs, Location unify) {
  //assert(false && "to be implemented");

  std::vector<CommitSite> commits;
  std::vector<InferenceSite> infers;

  Unification uni;
  return uni.unify(lhs, rhs, unify, true /*commit*/, true /*emit error*/,
                   false
                   /*infer*/,
                   commits, infers);
}

TyTy::BaseType *unifyWithSite(basic::NodeId, TyTy::WithLocation lhs,
                              TyTy::WithLocation rhs, Location unify) {
  std::vector<CommitSite> commits;
  std::vector<InferenceSite> infers;

  Unification uni;
  return uni.unify(lhs, rhs, unify, true /*commit*/, true /*emit error*/,
                   false
                   /*infer*/,
                   commits, infers);
}

} // namespace rust_compiler::sema::type_checking
