#include "Unification.h"

#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TyTy.h"

#include <vector>

using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::tyctx;

TyTy::BaseType *Unification::unify(bool forceCommit_, bool errors, bool infer) {

  TyTy::BaseType *leftType = lhs.getType();
  TyTy::BaseType *rightType = rhs.getType();
  inference = infer;
  emitErrors = errors;
  forceCommit = forceCommit_;

  assert(leftType->getNumberOfSpecifiedBounds() == 0);

  assert(inference == false);

  if (inference)
    handleInference();

  TyTy::BaseType *result = expect(leftType, rightType);

  commits.push_back(CommitSite{leftType, rightType, result});

  if (forceCommit)
    commit(leftType, rightType, result);

  if (result->getKind() == TypeKind::Error && emitErrors)
    emitFailedUnification();

  return result;
}

void Unification::handleInference() {
  assert(false);
  infers;
}

void Unification::emitFailedUnification() { assert(false); }

void Unification::commit(TyTy::BaseType *leftType, TyTy::BaseType *rightType,
                         TyTy::BaseType *result) {

  BaseType *b = leftType->destructure();
  BaseType *o = rightType->destructure();

  result->appendReference(b->getReference());
  result->appendReference(o->getReference());

  for (auto ref : b->getCombinedReferences())
    result->appendReference(ref);

  for (auto ref : o->getCombinedReferences())
    result->appendReference(ref);

  o->appendReference(result->getReference());
  o->appendReference(b->getReference());
  b->appendReference(result->getReference());
  b->appendReference(o->getReference());

  bool isResolved = result->getKind() != TypeKind::Inferred;
  bool isInfererenceVariable = result->getKind() == TypeKind::Inferred;
  bool isScalarInferenceVariable =
      isInfererenceVariable &&
      (static_cast<InferType *>(result)->getInferredKind() !=
       InferKind::General);

  if (isResolved || isScalarInferenceVariable) {

    for (NodeId ref : result->getCombinedReferences()) {
      std::optional<BaseType *> referenceType = context->lookupType(ref);
      if (!referenceType)
        continue;

      if ((*referenceType)->getKind() == TypeKind::Inferred)
        context->insertType(
            NodeIdentity(ref,
                         rust_compiler::session::session->getCurrentCrateNum(),
                         location),
            result->clone());
    }
  }
  assert(false);
}

TyTy::BaseType *Unification::expect(TyTy::BaseType *leftType,
                                    TyTy::BaseType *rightType) {
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
    return expectUSizeType(static_cast<TyTy::USizeType *>(leftType), rightType);
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
    return expectTuple(static_cast<TyTy::TupleType *>(leftType), rightType);
  }
  case TyTy::TypeKind::Parameter: {
    assert(false);
  }
  case TyTy::TypeKind::ADT: {
    assert(false);
  }
  case TyTy::TypeKind::Array: {
    assert(false);
  }
  case TyTy::TypeKind::Projection: {
    assert(false);
  }
  case TyTy::TypeKind::Error: {
    assert(false);
  }
  case TyTy::TypeKind::Slice: {
    assert(false);
  }
  case TyTy::TypeKind::Dynamic: {
    assert(false);
  }
  case TyTy::TypeKind::PlaceHolder: {
    assert(false);
  }
  case TyTy::TypeKind::FunctionPointer: {
    assert(false);
  }
  case TyTy::TypeKind::RawPointer: {
    assert(false);
  }
  case TyTy::TypeKind::Reference: {
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
  case TyTy::TypeKind::ADT: {
    assert(false);
  }
  case TyTy::TypeKind::Array: {
    assert(false);
  }
  case TyTy::TypeKind::Projection: {
    assert(false);
  }
  case TyTy::TypeKind::Slice: {
    assert(false);
  }
  case TyTy::TypeKind::Dynamic: {
    assert(false);
  }
  case TyTy::TypeKind::PlaceHolder: {
    assert(false);
  }
  case TyTy::TypeKind::FunctionPointer: {
    assert(false);
  }
  case TyTy::TypeKind::RawPointer: {
    assert(false);
  }
  case TyTy::TypeKind::Reference: {
    assert(false);
  }
  case TyTy::TypeKind::Error: {
    assert(false);
  }
  }
  assert(false);
}

TyTy::BaseType *Unification::expectUSizeType(TyTy::USizeType *left,
                                             TyTy::BaseType *right) {
  switch (right->getKind()) {
  case TyTy::TypeKind::Bool: {
    assert(false);
  }
  case TyTy::TypeKind::Char: {
    assert(false);
  }
  case TyTy::TypeKind::Int: {
    //    TyTy::IntType *rightInt = static_cast<TyTy::IntType *>(right);
    //    if (rightInt->getIntKind() == left->getIntKind())
    //      return new TyTy::IntType(left->getTypeReference(),
    //      left->getIntKind());
    assert(false);
  }
  case TyTy::TypeKind::Uint: {
    assert(false);
  }
  case TyTy::TypeKind::USize: {
    return right;
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
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(right);
    if (infer->getInferredKind() != TyTy::InferKind::Float) {
      infer->applyScalarTypeHint(*left);
      return infer;
    }
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
  case TyTy::TypeKind::ADT: {
    assert(false);
  }
  case TyTy::TypeKind::Array: {
    assert(false);
  }
  case TyTy::TypeKind::Projection: {
    assert(false);
  }
  case TyTy::TypeKind::Slice: {
    assert(false);
  }
  case TyTy::TypeKind::Dynamic: {
    assert(false);
  }
  case TyTy::TypeKind::PlaceHolder: {
    assert(false);
  }
  case TyTy::TypeKind::FunctionPointer: {
    assert(false);
  }
  case TyTy::TypeKind::RawPointer: {
    assert(false);
  }
  case TyTy::TypeKind::Reference: {
    assert(false);
  }
  case TyTy::TypeKind::Error: {
    assert(false);
  }
  }
  assert(false);
}

TyTy::BaseType *Unification::expectTuple(TyTy::TupleType *left,
                                         TyTy::BaseType *right) {
  switch (right->getKind()) {
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::USize:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::ADT:
  case TyTy::TypeKind::Slice:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::Array:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::FunctionPointer:
  case TyTy::TypeKind::RawPointer:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Error:
    return new TyTy::ErrorType(0);
  case TyTy::TypeKind::Inferred: {
    assert(false);
  }
  case TyTy::TypeKind::Tuple: {
    TyTy::TupleType *tuple = static_cast<TyTy::TupleType *>(right);
    if (left->getNumberOfFields() != tuple->getNumberOfFields())
      return new TyTy::ErrorType(0);
    // FIXME
    assert(false);
    std::vector<TyTy::TypeVariable> fields;
    for (size_t i = 0; i < left->getNumberOfFields(); ++i) {
      TyTy::BaseType *lel = left->getField(i);
      TyTy::BaseType *rel = tuple->getField(i);

      TyTy::BaseType *unifiedType = Unification::unifyWithSite(
          TyTy::WithLocation(lel), TyTy::WithLocation(rel), location, context);
      if (unifiedType->getKind() == TypeKind::Error)
        return new TyTy::ErrorType(0);

      fields.push_back(TypeVariable(unifiedType->getReference()));
    }

    return new TupleType(tuple->getReference(), Location::getEmptyLocation(),
                         fields);
  }
  }
}

TyTy::BaseType *Unification::unifyWithSite(TyTy::WithLocation lhs,
                                           TyTy::WithLocation rhs,
                                           Location unify, TyCtx *context) {

  std::vector<CommitSite> commits;
  std::vector<InferenceSite> infers;

  Unification uni = {commits, infers, unify, lhs, rhs, context};

  return uni.unify(true /*commit*/, true /*emitError*/, false /*infer*/);
}

} // namespace rust_compiler::sema::type_checking
