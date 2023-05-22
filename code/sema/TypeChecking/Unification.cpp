#include "Unification.h"

#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TyTy.h"
#include "llvm/Support/raw_ostream.h"

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

void Unification::emitFailedUnification() {

  llvm::errs() << rhs.getType()->toString() << "\n";
  llvm::errs() << lhs.getType()->toString() << "\n";

  assert(false);
}

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
    return expectUint(static_cast<TyTy::UintType *>(leftType), rightType);
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
    return expectInferenceVariable(static_cast<TyTy::InferType *>(leftType),
                                   rightType);
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
    return expectArray(static_cast<TyTy::ArrayType *>(leftType), rightType);
  }
  case TyTy::TypeKind::Projection: {
    assert(false);
  }
  case TyTy::TypeKind::Error: {
    assert(false);
  }
  case TyTy::TypeKind::Slice: {
    return expectSlice(static_cast<TyTy::SliceType *>(leftType), rightType);
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
    return expectRawPointer(static_cast<TyTy::RawPointerType *>(leftType),
                            rightType);
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
    break;
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
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(right);
    if (infer->getInferredKind() != TyTy::InferKind::Float) {
      infer->applyScalarTypeHint(*left);
      return left->clone();
      ;
    }
    assert(false);
  }
  case TyTy::TypeKind::USize: {
    return right->clone();
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Tuple:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::ADT:
  case TyTy::TypeKind::Array:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Slice:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::FunctionPointer:
  case TyTy::TypeKind::RawPointer:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  }
  }
  return new ErrorType(0);
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

    std::vector<TyTy::TypeVariable> fields;
    for (size_t i = 0; i < left->getNumberOfFields(); ++i) {
      TyTy::BaseType *bo = left->getField(i);
      TyTy::BaseType *fo = tuple->getField(i);

      TyTy::BaseType *unifiedType = Unification::unifyWithSite(
          TyTy::WithLocation(bo), TyTy::WithLocation(fo), location, context);

      // FIXME: commits, infers
      if (unifiedType->getKind() == TypeKind::Error)
        return new TyTy::ErrorType(0);

      fields.push_back(TypeVariable(unifiedType->getReference()));
    }

    return new TupleType(tuple->getReference(), tuple->getTypeReference(),
                         Location::getEmptyLocation(), fields);
    break;
  }
  }
}

TyTy::BaseType *
Unification::expectInferenceVariable(TyTy::InferType *left,
                                     TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TypeKind::Bool: {
    assert(false);
  }
  case TypeKind::Char: {
    assert(false);
  }
  case TypeKind::Int: {
    assert(false);
  }
  case TypeKind::Uint: {
    assert(false);
  }
  case TypeKind::USize: {
    assert(false);
  }
  case TypeKind::ISize: {
    assert(false);
  }
  case TypeKind::Float: {
    assert(false);
  }
  case TypeKind::Closure: {
    assert(false);
  }
  case TypeKind::Function: {
    assert(false);
  }
  case TypeKind::Inferred: {
    // TyTy::InferType *r = static_cast<TyTy::InferType*>(rightType);
    switch (left->getInferredKind()) {
    case InferKind::Integral: {
      assert(false);
    }
    case InferKind::Float: {
      assert(false);
    }
    case InferKind::General: {
      return rightType->clone();
    }
    }
    assert(false);
  }
  case TypeKind::Never: {
    assert(false);
  }
  case TypeKind::Str: {
    assert(false);
  }
  case TypeKind::Tuple: {
    assert(false);
  }
  case TypeKind::Parameter: {
    assert(false);
  }
  case TypeKind::ADT: {
    assert(false);
  }
  case TypeKind::Array: {
    assert(false);
  }
  case TypeKind::Slice: {
    assert(false);
  }
  case TypeKind::Projection: {
    assert(false);
  }
  case TypeKind::Dynamic: {
    assert(false);
  }
  case TypeKind::PlaceHolder: {
    assert(false);
  }
  case TypeKind::FunctionPointer: {
    assert(false);
  }
  case TypeKind::RawPointer: {
    assert(false);
  }
  case TypeKind::Reference: {
    assert(false);
  }
  case TypeKind::Error: {
    assert(false);
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

TyTy::BaseType *Unification::expectRawPointer(TyTy::RawPointerType *pointer,
                                              TyTy::BaseType *rightType) {

  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General) {
      return pointer->clone();
    }
    return new ErrorType(0);
  }
  case TyTy::TypeKind::USize:
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Tuple:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::ADT:
  case TyTy::TypeKind::Array:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Slice:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::FunctionPointer:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  case TyTy::TypeKind::RawPointer:
    TyTy::RawPointerType *type = static_cast<TyTy::RawPointerType *>(rightType);

    TyTy::BaseType *baseType = pointer->getBase();
    TyTy::BaseType *otherBaseType = type->getBase();

    TyTy::BaseType *resolved = Unification::unifyWithSite(
        TyTy::WithLocation(baseType), TyTy::WithLocation(otherBaseType),
        location, context);

    if (resolved->getKind() == TypeKind::Error)
      return new ErrorType(0);

    assert(false);
  }
  return new ErrorType(0);
}

TyTy::BaseType *Unification::expectSlice(TyTy::SliceType *leftType,
                                         TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General) {
      return leftType->clone();
    }
    return new ErrorType(0);
  }
  case TyTy::TypeKind::USize:
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Tuple:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::ADT:
  case TyTy::TypeKind::Array:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::FunctionPointer:
  case TyTy::TypeKind::RawPointer:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  case TyTy::TypeKind::Slice: {
    TyTy::SliceType *type = static_cast<TyTy::SliceType *>(rightType);
    TyTy::BaseType *elementUnify = Unification::unifyWithSite(
        TyTy::WithLocation(leftType->getElementType()),
        TyTy::WithLocation(type->getElementType()), location, context);
    if (elementUnify->getKind() == TypeKind::Error)
      return new ErrorType(0);
    return new TyTy::SliceType(
        type->getReference(), type->getTypeReference(),
        type->getTypeIdentity().getLocation(),
        TyTy::TypeVariable(elementUnify->getReference()));
  }
  }
  return new ErrorType(0);
}

TyTy::BaseType *Unification::expectArray(TyTy::ArrayType *array,
                                         TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General) {
      return array->clone();
    }
    return new ErrorType(0);
  }
  case TyTy::TypeKind::USize:
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Tuple:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::ADT:
  case TyTy::TypeKind::Slice:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::FunctionPointer:
  case TyTy::TypeKind::RawPointer:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  case TyTy::TypeKind::Array: {
    TyTy::ArrayType *type = static_cast<TyTy::ArrayType *>(rightType);
    TyTy::BaseType *elementUnify = Unification::unifyWithSite(
        TyTy::WithLocation(array->getElementType()),
        TyTy::WithLocation(type->getElementType()), location, context);
    if (elementUnify->getKind() == TypeKind::Error)
      return new ErrorType(0);
    return new TyTy::ArrayType(
        type->getReference(), type->getTypeIdentity().getLocation(),
        type->getCapacityExpression(),
        TyTy::TypeVariable(elementUnify->getReference()));
  }
  }
  return new ErrorType(0);
}

TyTy::BaseType *Unification::expectUint(TyTy::UintType *leftType,
                                        TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General or
        infer->getInferredKind() == TyTy::InferKind::Integral) {
      infer->applyScalarTypeHint(*leftType);
      return leftType->clone();
    }
    return new ErrorType(0);
  }
  case TyTy::TypeKind::USize:
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Int:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::Never:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Tuple:
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
    return new ErrorType(0);
  case TyTy::TypeKind::Uint:
    TyTy::UintType *type = static_cast<TyTy::UintType *>(rightType);
    if (leftType->getUintKind() == type->getUintKind())
      return new TyTy::UintType(type->getReference(), type->getTypeReference(),
                                type->getUintKind());
    return new ErrorType(0);
  }
  return new ErrorType(0);
}

} // namespace rust_compiler::sema::type_checking
