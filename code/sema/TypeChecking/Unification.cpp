#include "Unification.h"

#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TyTy.h"

#include <ios>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
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

  llvm::errs() << "@" << location.toString() << "\n";
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
    return expectStr(static_cast<TyTy::StrType *>(leftType), rightType);
  }
  case TyTy::TypeKind::Tuple: {
    return expectTuple(static_cast<TyTy::TupleType *>(leftType), rightType);
  }
  case TyTy::TypeKind::Parameter: {
    assert(false);
  }
  case TyTy::TypeKind::ADT: {
    return expectADT(static_cast<TyTy::ADTType *>(leftType), rightType);
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
    return expectReference(static_cast<TyTy::ReferenceType *>(leftType),
                           rightType);
  }
  }
}

TyTy::BaseType *Unification::expectIntType(TyTy::IntType *left,
                                           TyTy::BaseType *right) {
  switch (right->getKind()) {
  case TyTy::TypeKind::Int: {
    TyTy::IntType *rightInt = static_cast<TyTy::IntType *>(right);
    if (rightInt->getIntKind() == left->getIntKind())
      return new TyTy::IntType(rightInt->getReference(),
                               rightInt->getTypeReference(),
                               rightInt->getIntKind());
    return new ErrorType(0);
  }
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(right);
    if (infer->getInferredKind() == TyTy::InferKind::Integral ||
        infer->getInferredKind() == InferKind::General) {
      infer->applyScalarTypeHint(*left);
      return left->clone();
    }
    return new ErrorType(0);
  }
  case TyTy::TypeKind::Bool:
  case TyTy::TypeKind::Char:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::ISize:
  case TyTy::TypeKind::Float:
  case TyTy::TypeKind::Closure:
  case TyTy::TypeKind::Function:
  case TyTy::TypeKind::USize:
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
  llvm_unreachable("all cases covered");
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
    TyTy::InferType *r = static_cast<TyTy::InferType *>(right);
    if (r->getInferredKind() == InferKind::General)
      return left->clone();
    return new TyTy::ErrorType(0);
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
  }
  }
}

TyTy::BaseType *
Unification::expectInferenceVariable(TyTy::InferType *left,
                                     TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TypeKind::Inferred: {
    TyTy::InferType *r = static_cast<TyTy::InferType *>(rightType);
    switch (left->getInferredKind()) {
    case InferKind::Integral: {
      if (r->getInferredKind() == InferKind::Integral ||
          r->getInferredKind() == InferKind::General)
        return rightType->clone();
      return new TyTy::ErrorType(0);
    }
    case InferKind::Float: {
      if (r->getInferredKind() == InferKind::Float ||
          r->getInferredKind() == InferKind::General)
        return rightType->clone();
      return new TyTy::ErrorType(0);
    }
    case InferKind::General: {
      return rightType->clone();
    }
    }
    return new ErrorType(0);
  }
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::ISize: {
    if (left->getInferredKind() == InferKind::General ||
        left->getInferredKind() == InferKind::Integral) {
      left->applyScalarTypeHint(*rightType);
      return rightType->clone();
    }
    return new ErrorType(0);
  }
  case TypeKind::Float: {
    if (left->getInferredKind() == InferKind::General ||
        left->getInferredKind() == InferKind::Float) {
      left->applyScalarTypeHint(*rightType);
      return rightType->clone();
    }
    return new ErrorType(0);
  }
  case TypeKind::ADT:
  case TypeKind::Str:
  case TypeKind::Reference:
  case TypeKind::RawPointer:
  case TypeKind::Parameter:
  case TypeKind::Array:
  case TypeKind::Slice:
  case TypeKind::Function:
  case TypeKind::FunctionPointer:
  case TypeKind::Tuple:
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Never:
  case TypeKind::PlaceHolder:
  case TypeKind::Projection:
  case TypeKind::Dynamic:
  case TypeKind::Closure: {
    if (left->getInferredKind() == InferKind::General)
      return rightType->clone();
    return new ErrorType(0);
  }
  case TypeKind::Error:
    return new ErrorType(0);
  }
  llvm::llvm_unreachable_internal("all cases covered");
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
  case TyTy::TypeKind::RawPointer: {
    TyTy::RawPointerType *type = static_cast<TyTy::RawPointerType *>(rightType);

    TyTy::BaseType *baseType = pointer->getBase();
    TyTy::BaseType *otherBaseType = type->getBase();

    TyTy::BaseType *resolved = Unification::unifyWithSite(
        TyTy::WithLocation(baseType), TyTy::WithLocation(otherBaseType),
        location, context);

    if (resolved->getKind() == TypeKind::Error)
      return new ErrorType(0);

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
  }
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

TyTy::BaseType *Unification::expectReference(TyTy::ReferenceType *leftType,
                                             TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General)
      return leftType->clone();
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
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  case TyTy::TypeKind::Reference:
    TyTy::ReferenceType *ref = static_cast<TyTy::ReferenceType *>(rightType);
    TyTy::BaseType *resolvedType = Unification::unifyWithSite(
        TyTy::WithLocation(leftType->getBase()),
        TyTy::WithLocation(ref->getBase()), location, context);
    if (resolvedType->getKind() == TypeKind::Error)
      return new ErrorType(0);
    bool acceptableMutability = leftType->isMutable() ? ref->isMutable() : true;
    if (!acceptableMutability)
      return new ErrorType(0);

    return new ReferenceType(
        leftType->getReference(), leftType->getTypeReference(),
        TyTy::TypeVariable(resolvedType->getReference()), leftType->getMut());
  }
  return new ErrorType(0);
}

TyTy::BaseType *Unification::expectStr(TyTy::StrType *leftType,
                                       TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General)
      return leftType->clone();
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
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  case TyTy::TypeKind::Str:
    return rightType->clone();
  }
  return new ErrorType(0);
}

TyTy::BaseType *Unification::expectADT(TyTy::ADTType *left,
                                       TyTy::BaseType *rightType) {
  switch (rightType->getKind()) {
  case TyTy::TypeKind::Inferred: {
    TyTy::InferType *infer = static_cast<TyTy::InferType *>(rightType);
    if (infer->getInferredKind() == TyTy::InferKind::General)
      return left->clone();
    return new ErrorType(0);
  }
  case TyTy::TypeKind::ADT: {
    TyTy::ADTType *right = static_cast<TyTy::ADTType *>(rightType);
    if (left->getKind() != right->getKind())
      return new ErrorType(0);
    if (left->getIdentifier() != right->getIdentifier())
      return new ErrorType(0);
    if (left->getNumberOfVariants() != right->getNumberOfVariants())
      return new ErrorType(0);
    for (size_t i = 0; i < left->getNumberOfVariants(); ++i) {
      TyTy::VariantDef *a = left->getVariant(i);
      TyTy::VariantDef *b = right->getVariant(i);

      if (a->getNumberOfFields() != b->getNumberOfFields())
        return new ErrorType(0);

      for (size_t j = 0; i < a->getNumberOfFields(); ++j) {
        TyTy::StructFieldType *baseField = a->getFieldAt(j);
        TyTy::StructFieldType *otherField = b->getFieldAt(j);

        TyTy::BaseType *baseFieldType = baseField->getFieldType();
        TyTy::BaseType *otherFieldType = otherField->getFieldType();

        TyTy::BaseType *unifiedType = Unification::unifyWithSite(
            TyTy::WithLocation(baseFieldType),
            TyTy::WithLocation(otherFieldType), location, context);

        if (unifiedType->getKind() == TypeKind::Error)
          return new ErrorType(0);
      }
    }

    if (right->isUnit() && left->isUnit()) {
      if (right->getNumberOfSubstitutions() != left->getNumberOfSubstitutions())
        return new ErrorType(0);

      for (size_t i = 0; i < right->getNumberOfSubstitutions(); ++i) {
        SubstitutionParamMapping &a = left->getSubstitutions()[i];
        SubstitutionParamMapping &b = right->getSubstitutions()[i];

        ParamType *pa = a.getParamType();
        ParamType *pb = b.getParamType();

        TyTy::BaseType *result = Unification::unifyWithSite(
            TyTy::WithLocation(pa), TyTy::WithLocation(pb), location, context);

        if (result->getKind() == TypeKind::Error)
          return new TyTy::ErrorType(0);
      }
    }

    return left->clone();
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
  case TyTy::TypeKind::Tuple:
  case TyTy::TypeKind::Parameter:
  case TyTy::TypeKind::Slice:
  case TyTy::TypeKind::Projection:
  case TyTy::TypeKind::Dynamic:
  case TyTy::TypeKind::Array:
  case TyTy::TypeKind::PlaceHolder:
  case TyTy::TypeKind::FunctionPointer:
  case TyTy::TypeKind::RawPointer:
  case TyTy::TypeKind::Uint:
  case TyTy::TypeKind::Reference:
  case TyTy::TypeKind::Str:
  case TyTy::TypeKind::Error:
    return new ErrorType(0);
  }
  llvm_unreachable("all cases covered");
}

} // namespace rust_compiler::sema::type_checking
