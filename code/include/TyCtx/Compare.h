#pragma once

#include "Session/Session.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"
#include <llvm/Support/ErrorHandling.h>

namespace rust_compiler::tyctx::TyTy {

class BaseCmp {

public:
  virtual ~BaseCmp() = default;
  virtual bool canEqual(const BaseType *other) = 0;
};

class ADTCmp : public BaseCmp {
  const ADTType *it;

  bool visitInfer(const BaseType *other) {
    const InferType *infer = static_cast<const InferType *>(other);
    return infer->getInferredKind() == InferKind::General;
  }

  bool visitADT(const BaseType *other) {
    const ADTType *foo = static_cast<const ADTType *>(it);
    const ADTType *bar = static_cast<const ADTType *>(other);

    if (foo->getKind() != bar->getKind())
      return false;

    if (foo->getIdentifier() != bar->getIdentifier())
      return false;

    if (foo->getNumberOfVariants() != bar->getNumberOfVariants())
      return false;

    for (size_t i = 0; i < bar->getNumberOfVariants(); ++i) {
      TyTy::VariantDef *a = foo->getVariant(i);
      TyTy::VariantDef *b = bar->getVariant(i);

      if (a->getFields() != b->getFields())
        return false;

      for (size_t j = 0; i < a->getNumberOfFields(); ++j) {
        TyTy::StructFieldType *baseField = a->getFieldAt(j);
        TyTy::StructFieldType *otherField = b->getFieldAt(j);

        TyTy::BaseType *baseFieldType = baseField->getFieldType();
        TyTy::BaseType *otherFieldType = otherField->getFieldType();

        if (!baseFieldType - canEqual(otherFieldType))
          return false;
      }
    }
    return true;
  }

public:
  ADTCmp(const ADTType *it) : it(it) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::FunctionPointer:
    case TypeKind::Closure:
      return false;
    case TypeKind::ADT:
      return visitADT(other);
    case TypeKind::Inferred:
      return visitInfer(other);
    }
  }
};

class IntCmp : public BaseCmp {
  const IntType *it;

public:
  IntCmp(const IntType *it) : it(it) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
      return false;
    case TypeKind::Int:
      return it->getIntKind() ==
             static_cast<const IntType *>(other)->getIntKind();
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() !=
             InferKind::Float;
    }
  }
};

class UintCmp : public BaseCmp {
  const UintType *it;

public:
  UintCmp(const UintType *it) : it(it) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
      return false;
    case TypeKind::Uint:
      return it->getUintKind() ==
             static_cast<const UintType *>(other)->getUintKind();
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() !=
             InferKind::Float;
    }
  }
};

class FloatCmp : public BaseCmp {
  const FloatType *it;

public:
  FloatCmp(const FloatType *it) : it(it) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
      return false;
    case TypeKind::Float:
      return it->getFloatKind() ==
             static_cast<const FloatType *>(other)->getFloatKind();
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() !=
             InferKind::Integral;
    }
  }
};

class TupleCmp : public BaseCmp {
  const TupleType *it;
  bool emitErrors;

  bool visitTuple(const TupleType *other) {
    if (it->getNumberOfFields() != other->getNumberOfFields())
      return false;

    for (size_t i = 0; i < it->getNumberOfFields(); ++i) {
      BaseType *bo = it->getField(i);
      BaseType *fo = other->getField(i);

      return bo->canEqual(fo, emitErrors);
    }

    return true;
  }

public:
  TupleCmp(const TupleType *it, bool emitErrors)
      : it(it), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::Float:
      return false;
    case TypeKind::Tuple:
      return visitTuple(static_cast<const TupleType *>(other));
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() !=
             InferKind::Integral;
    }
  }
};

class USizeCmp : public BaseCmp {

public:
  USizeCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::ISize:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
      return false;
    case TypeKind::USize:
      return true;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() !=
             InferKind::Float;
    }
  }
};

class ISizeCmp : public BaseCmp {

public:
  ISizeCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
      return false;
    case TypeKind::ISize:
      return true;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() !=
             InferKind::Float;
    }
  }
};

class CharCmp : public BaseCmp {

public:
  CharCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
      return false;
    case TypeKind::Char:
      return true;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    }
  }
};

class ReferenceCmp : public BaseCmp {
  const ReferenceType *it;
  bool emitErrors;

  bool visitReference(const BaseType *other) {
    const ReferenceType *ref = static_cast<const ReferenceType *>(other);

    BaseType *baseType = it->getBase();
    BaseType *otherBaseType = ref->getBase();

    if (it->isMutable() != ref->isMutable())
      return false;

    if (!baseType->canEqual(otherBaseType, emitErrors))
      return false;

    return true;
  }

public:
  ReferenceCmp(const ReferenceType *it, bool emitErrors)
      : it(it), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
      return false;
    case TypeKind::Reference:
      return visitReference(other);
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    }
  }
};

class RawPointerCmp : public BaseCmp {
  const RawPointerType *it;
  bool emitErrors;

  bool visitRawPointer(const BaseType *other) {

    const RawPointerType *oth = static_cast<const RawPointerType *>(other);

    BaseType *baseType = it->getBase();
    BaseType *otherBaseType = oth->getBase();

    if (it->isMutable() != oth->isMutable())
      return false;

    if (!baseType->canEqual(otherBaseType, emitErrors))
      return false;

    return true;
  }

public:
  RawPointerCmp(const RawPointerType *it, bool emitErrors)
      : it(it), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
      return false;
    case TypeKind::RawPointer:
      return visitRawPointer(other);
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    }
  }
};

class ParamCmp : public BaseCmp {
  const ParamType *it;

public:
  ParamCmp(const ParamType *it) : it(it) {}

  bool canEqual(const BaseType *other) override {

    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Inferred:
      return true;
    case TypeKind::PlaceHolder:
      return it->getSymbol() ==
             static_cast<const ParamType *>(other)->getSymbol();
    }
  }
};

class StrCmp : public BaseCmp {

public:
  StrCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
      return false;
    case TypeKind::Str:
      return true;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    }
  }
};

class NeverCmp : public BaseCmp {

public:
  NeverCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
      return false;
    case TypeKind::Never:
      return true;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    }
  }
};

class PlaceholderCmp : public BaseCmp {

public:
  PlaceholderCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
    case TypeKind::Never:
      return true;
    case TypeKind::PlaceHolder:
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    }
  }
};

class InferCmp : public BaseCmp {
  const InferType *base;

public:
  InferCmp(const InferType *base) : base(base) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Float:
      return (base->getInferredKind() == InferKind::General) or
             (base->getInferredKind() == InferKind::Float);
    case TypeKind::Int:
      return (base->getInferredKind() == InferKind::General) or
             (base->getInferredKind() == InferKind::Integral);
    case TypeKind::Uint:
      return (base->getInferredKind() == InferKind::General) or
             (base->getInferredKind() == InferKind::Integral);
    case TypeKind::USize:
      return (base->getInferredKind() == InferKind::General) or
             (base->getInferredKind() == InferKind::Integral);
    case TypeKind::ISize:
      return (base->getInferredKind() == InferKind::General) or
             (base->getInferredKind() == InferKind::Integral);
    case TypeKind::Str:
    case TypeKind::Error:
    case TypeKind::FunctionPointer:
    case TypeKind::Projection:
    case TypeKind::Function:
      return false;
    case TypeKind::PlaceHolder:
    case TypeKind::Inferred: {
      switch (base->getInferredKind()) {
      case InferKind::General:
        return true;
      case InferKind::Integral: {
        return (base->getInferredKind() == InferKind::General) or
               (base->getInferredKind() == InferKind::Integral);
      case InferKind::Float: {
        return (base->getInferredKind() == InferKind::General) or
               (base->getInferredKind() == InferKind::Float);
      }
      }
      }
      llvm_unreachable("all cases covered");
    }
    case TypeKind::Bool:
    case TypeKind::Array:
    case TypeKind::Slice:
    case TypeKind::ADT:
    case TypeKind::Tuple:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::Dynamic:
    case TypeKind::Closure:
    case TypeKind::RawPointer:
      return base->getInferredKind() == InferKind::General;
    case TypeKind::Parameter:
    case TypeKind::Never:
      return true;
    }
  }
};

class FunctionCmp : public BaseCmp {
  const FunctionType *base;
  bool emitErrors;

  bool visitFunction(const FunctionType *other) {
    if (base->getNumberOfArguments() != other->getNumberOfArguments())
      return false;

    for (size_t i = 0; i < other->getNumberOfArguments(); ++i)
      if (!(base->getParameter(i).second)
               ->canEqual(other->getParameter(i).second, emitErrors))
        return false;

    if (!base->getReturnType()->canEqual(other->getReturnType(), emitErrors))
      return false;

    return true;
  }

public:
  FunctionCmp(const FunctionType *base, bool emitErrors)
      : base(base), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
    case TypeKind::Never:
    case TypeKind::PlaceHolder:
      return false;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    case TypeKind::Function:
      return visitFunction(static_cast<const FunctionType *>(other));
    }
  }
};

class FunctionPointerCmp : public BaseCmp {
  const FunctionPointerType *base;
  bool emitErrors;

  bool visitFunction(const FunctionType *other) {
    if (base->getNumberOfArguments() != other->getNumberOfArguments())
      return false;

    for (size_t i = 0; i < other->getNumberOfArguments(); ++i)
      if (!(base->getParameter(i))
               ->canEqual(other->getParameter(i).second, emitErrors))
        return false;

    if (!base->getReturnType()->canEqual(other->getReturnType(), emitErrors))
      return false;

    return true;
  }

  bool visitFunctionPointer(const FunctionPointerType *other) {
    if (base->getNumberOfArguments() != other->getNumberOfArguments())
      return false;

    for (size_t i = 0; i < other->getNumberOfArguments(); ++i)
      if (!base->getParameter(i)->canEqual(other->getParameter(i), emitErrors))
        return false;

    if (!base->getReturnType()->canEqual(other->getReturnType(), emitErrors))
      return false;

    return true;
  }

public:
  FunctionPointerCmp(const FunctionPointerType *base, bool emitErrors)
      : base(base), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
    case TypeKind::Never:
    case TypeKind::PlaceHolder:
      return false;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    case TypeKind::Function:
      return visitFunction(static_cast<const FunctionType *>(other));
    case TypeKind::FunctionPointer:
      return visitFunctionPointer(
          static_cast<const FunctionPointerType *>(other));
    }
  }
};

class ClosureCmp : public BaseCmp {
  const ClosureType *base;
  bool emitErrors;

  bool visitClosure(const ClosureType *other) {
    if (base->getTypeReference() != other->getTypeReference())
      return false;

    if (!base->getParameters()->canEqual(other->getParameters(), false))
      return false;

    if (!base->getResultType()->canEqual(other->getResultType(), emitErrors))
      return false;

    return true;
  }

public:
  ClosureCmp(const ClosureType *base, bool emitErrors)
      : base(base), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::FunctionPointer:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
    case TypeKind::Never:
    case TypeKind::Function:
    case TypeKind::PlaceHolder:
      return false;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    case TypeKind::Closure:
      return visitClosure(static_cast<const ClosureType *>(other));
    }
  }
};

class ArrayCmp : public BaseCmp {
  const ArrayType *base;
  bool emitErrors;

  bool visitArray(const ArrayType *other) {
    if (!base->getElementType()->canEqual(other->getElementType(), emitErrors))
      return false;

    return true;
  }

public:
  ArrayCmp(const ArrayType *base, bool emitErrors)
      : base(base), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::FunctionPointer:
    case TypeKind::Closure:
    case TypeKind::Slice:
    case TypeKind::Projection:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
    case TypeKind::Never:
    case TypeKind::Function:
    case TypeKind::PlaceHolder:
      return false;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    case TypeKind::Array:
      return visitArray(static_cast<const ArrayType *>(other));
    }
  }
};

class SliceCmp : public BaseCmp {
  const SliceType *base;
  bool emitErrors;

  bool visitSlice(const SliceType *other) {
    if (!base->getElementType()->canEqual(other->getElementType(), emitErrors))
      return false;

    return true;
  }

public:
  SliceCmp(const SliceType *base, bool emitErrors)
      : base(base), emitErrors(emitErrors) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Closure:
    case TypeKind::Float:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::FunctionPointer:
    case TypeKind::Projection:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Char:
    case TypeKind::Reference:
    case TypeKind::RawPointer:
    case TypeKind::Str:
    case TypeKind::Never:
    case TypeKind::Function:
    case TypeKind::PlaceHolder:
    case TypeKind::Array:
      return false;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    case TypeKind::Slice:
      return visitSlice(static_cast<const SliceType *>(other));
    }
  }
};

class BoolCmp : public BaseCmp {

public:
  BoolCmp() {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Char:
    case TypeKind::Uint:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Float:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::Dynamic:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::Int:
    case TypeKind::ADT:
      return false;
    case TypeKind::Inferred:
      return static_cast<const InferType *>(other)->getInferredKind() ==
             InferKind::General;
    case TypeKind::Bool:
      return true;
    }
  }
};

class DynamicCmp : public BaseCmp {
  const DynamicObjectType *it;

  bool visitDynamic(const DynamicObjectType *other) {
    if (it->getNumberOfSpecifiedBounds() != other->getNumberOfSpecifiedBounds())
      return false;

    tyctx::TyCtx *tcx = rust_compiler::session::session->getTypeContext();
    std::optional<Location> loc = tcx->lookupLocation(other->getReference());
    assert(loc.has_value());
    return it->boundsCompatible(other, *loc, false);
  }

public:
  DynamicCmp(const DynamicObjectType *it) : it(it) {}

  bool canEqual(const BaseType *other) override {
    switch (other->getKind()) {
    case TypeKind::Bool:
    case TypeKind::Char:
    case TypeKind::USize:
    case TypeKind::ISize:
    case TypeKind::Never:
    case TypeKind::Str:
    case TypeKind::Tuple:
    case TypeKind::Parameter:
    case TypeKind::Array:
    case TypeKind::Error:
    case TypeKind::PlaceHolder:
    case TypeKind::FunctionPointer:
    case TypeKind::RawPointer:
    case TypeKind::Slice:
    case TypeKind::Reference:
    case TypeKind::Projection:
    case TypeKind::Function:
    case TypeKind::Closure:
    case TypeKind::ADT:
    case TypeKind::Int:
    case TypeKind::Float:
    case TypeKind::Inferred:
    case TypeKind::Uint:
      return false;
    case TypeKind::Dynamic:
      return visitDynamic(static_cast<const DynamicObjectType *>(other));
    }
  }
};

} // namespace rust_compiler::tyctx::TyTy
