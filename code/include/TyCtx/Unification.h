#pragma once

#include "Location.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"

#include <vector>

namespace rust_compiler::tyctx {

using namespace rust_compiler::tyctx;

class CommitSite {
public:
  CommitSite(TyTy::BaseType *lhs, TyTy::BaseType *rhs, TyTy::BaseType *resolved)
      : lhs(lhs), rhs(rhs), resolved(resolved) {}

private:
  [[maybe_unused]] TyTy::BaseType *lhs;
  [[maybe_unused]] TyTy::BaseType *rhs;
  [[maybe_unused]] TyTy::BaseType *resolved;
};

class InferenceSite {};

class Unification {
public:
  Unification(std::vector<CommitSite> &commits,
              std::vector<InferenceSite> &infers, Location loc,
              TyTy::WithLocation lhs, TyTy::WithLocation rhs, TyCtx *context)
      : commits(commits), infers(infers), location(loc), lhs(lhs), rhs(rhs),
        context(context) {}

  static TyTy::BaseType *unifyWithSite(TyTy::WithLocation lhs,
                                       TyTy::WithLocation rhs, Location unify,
                                       TyCtx *);

  // @private
  TyTy::BaseType *unify(bool commit, bool emitErrors, bool inference);

private:
  TyTy::BaseType *expectIntType(TyTy::IntType *left, TyTy::BaseType *right);
  TyTy::BaseType *expectUSizeType(TyTy::USizeType *left, TyTy::BaseType *right);
  TyTy::BaseType *expectISizeType(TyTy::ISizeType *left, TyTy::BaseType *right);
  TyTy::BaseType *expectTuple(TyTy::TupleType *left, TyTy::BaseType *right);
  TyTy::BaseType *expectInferenceVariable(TyTy::InferType *left,
                                          TyTy::BaseType *rightType);
  TyTy::BaseType *expectRawPointer(TyTy::RawPointerType *,
                                   TyTy::BaseType *rightType);
  TyTy::BaseType *expectSlice(TyTy::SliceType *leftType,
                              TyTy::BaseType *rightType);
  TyTy::BaseType *expectArray(TyTy::ArrayType *, TyTy::BaseType *rightType);

  TyTy::BaseType *expect(TyTy::BaseType *left, TyTy::BaseType *right);
  TyTy::BaseType *expectUint(TyTy::UintType *leftType,
                             TyTy::BaseType *rightType);
  TyTy::BaseType *expectReference(TyTy::ReferenceType *,
                                  TyTy::BaseType *rightType);
  TyTy::BaseType *expectStr(TyTy::StrType *, TyTy::BaseType *rightType);
  TyTy::BaseType *expectADT(TyTy::ADTType *, TyTy::BaseType *rightType);
  TyTy::BaseType *expectNever(TyTy::NeverType *leftType,
                              TyTy::BaseType *rightType);

  void commit(TyTy::BaseType *leftType, TyTy::BaseType *rightType,
              TyTy::BaseType *result);

  void emitFailedUnification();

  void handleInference();

  std::vector<CommitSite> &commits;
  std::vector<InferenceSite> &infers;
  Location location;
  bool inference;
  bool emitErrors;
  bool forceCommit;

  TyTy::WithLocation lhs;
  TyTy::WithLocation rhs;
  TyCtx *context;
};

} // namespace rust_compiler::tyctx
