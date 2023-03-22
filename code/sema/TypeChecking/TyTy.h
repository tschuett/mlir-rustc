#pragma once

#include "Basic/Ids.h"
#include "Location.h"
#include "TypeIdentity.h"

namespace rust_compiler::sema::type_checking::TyTy {

/// https://doc.rust-lang.org/reference/types.html
// https://rustc-dev-guide.rust-lang.org/type-inference.html#inference-variables
// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html

enum class InferKind { Integral, Float, General };

enum class TypeKind {
  Bool,
  Char,
  Int,
  Uint,
  USize,
  ISize,
  Float,
  Closure,
  Function,
  Inferred,
  Never,
  Str,
  Tuple,

  Error
};

enum class IntKind { I8, I16, I32, I64, I128 };

enum class UintKind { U8, U16, U32, U64, U128 };

enum class FloatKind { F32, F64 };

class BaseType { // : public TypeBoundsMapping
public:
  virtual ~BaseType();

  basic::NodeId getReference() const;
  basic::NodeId getTypeReference() const;

  void setReference(basic::NodeId);

  TypeKind getKind() const { return kind; }

  virtual bool needsGenericSubstitutions() const = 0;

protected:
  BaseType(basic::NodeId ref, basic::NodeId ty_ref, TypeKind kind,
           TypeIdentity ident);

private:
  basic::NodeId reference;
  basic::NodeId typeReference;
  TypeKind kind;
  TypeIdentity identity;
};

class IntType : public BaseType {
public:
  IntType(basic::NodeId, IntKind);

  bool needsGenericSubstitutions() const override;

private:
  IntKind kind;
};

class UintType : public BaseType {
public:
  UintType(basic::NodeId, UintKind);

  bool needsGenericSubstitutions() const override;

private:
  UintKind kind;
};

class USizeType : public BaseType {
public:
  USizeType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

class ISizeType : public BaseType {
public:
  ISizeType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

class FloatType : public BaseType {
public:
  FloatType(basic::NodeId, FloatKind);

  bool needsGenericSubstitutions() const override;

private:
  FloatKind kind;
};

/// false or trur
class BoolType : public BaseType {
public:
  BoolType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

class CharType : public BaseType {
public:
  CharType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

class StrType : public BaseType {
public:
  StrType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

/// !
class NeverType : public BaseType {
public:
  NeverType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

/// (.., .., ..)
class TupleType : public BaseType {
public:
  TupleType(basic::NodeId, Location loc);

  bool needsGenericSubstitutions() const override;

  static TupleType *getUnitType(basic::NodeId);
};

class FunctionType : public BaseType {
public:
  FunctionType(basic::NodeId, Location loc);
};

class ClosureType : public BaseType {
public:
  ClosureType(basic::NodeId, Location loc);
};

class InferType : public BaseType {
public:
  InferType(basic::NodeId, Location loc);

  InferKind getInferredKind() const { return inferKind; }

  bool needsGenericSubstitutions() const override;

private:
  InferKind inferKind;
};

class ErrorType : public BaseType {
public:
  ErrorType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
};

class WithLocation {
public:
  WithLocation(BaseType *type, Location loc);
  WithLocation(BaseType *type);
};

} // namespace rust_compiler::sema::type_checking::TyTy
