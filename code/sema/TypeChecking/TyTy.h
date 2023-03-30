#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "Basic/Ids.h"
#include "Location.h"
#include "Substitutions.h"
#include "TyCtx/ItemIdentity.h"
#include "TyCtx/NodeIdentity.h"
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
  Parameter,

  Error
};

enum class IntKind { I8, I16, I32, I64, I128 };

enum class UintKind { U8, U16, U32, U64, U128 };

enum class FloatKind { F32, F64 };

class BaseType;

class TypeVariable {
public:
  TypeVariable(basic::NodeId id);
  TyTy::BaseType *getType() const;

private:
  basic::NodeId id;
};

class BaseType { // : public TypeBoundsMapping
public:
  virtual ~BaseType();

  basic::NodeId getReference() const;
  basic::NodeId getTypeReference() const;

  void setReference(basic::NodeId);

  TypeKind getKind() const { return kind; }

  virtual bool needsGenericSubstitutions() const = 0;
  virtual std::string toString() const = 0;

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
  std::string toString() const override;

private:
  IntKind kind;
};

class UintType : public BaseType {
public:
  UintType(basic::NodeId, UintKind);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

private:
  UintKind kind;
};

class USizeType : public BaseType {
public:
  USizeType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

class ISizeType : public BaseType {
public:
  ISizeType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

class FloatType : public BaseType {
public:
  FloatType(basic::NodeId, FloatKind);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

private:
  FloatKind kind;
};

/// false or true
class BoolType : public BaseType {
public:
  BoolType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

class CharType : public BaseType {
public:
  CharType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

class StrType : public BaseType {
public:
  StrType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

/// !
class NeverType : public BaseType {
public:
  NeverType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

/// (.., .., ..)
class TupleType : public BaseType {
public:
  TupleType(basic::NodeId, Location loc);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

  static TupleType *getUnitType(basic::NodeId);

private:
  std::vector<TypeVariable> fields;
};

class FunctionType : public BaseType {
public:
  FunctionType(
      basic::NodeId, std::string_view name, tyctx::ItemIdentity,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::vector<TyTy::SubstitutionParamMapping> substitutions);

  std::string toString() const override;

  TyTy::BaseType *getReturnType() const;

  bool needsGenericSubstitutions() const override;

private:
  basic::NodeId id;
  std::string name;
  tyctx::ItemIdentity ident;
  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                TyTy::BaseType *>>
      parameters;
  TyTy::BaseType *returnType = nullptr;
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
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
  std::string toString() const override;

private:
  InferKind inferKind;
};

class ErrorType : public BaseType {
public:
  ErrorType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
};

class WithLocation {
public:
  WithLocation(BaseType *type, Location loc) : type(type), loc(loc) {}
  WithLocation(BaseType *type)
      : type(type), loc(Location::getEmptyLocation()) {}

  BaseType *getType() const { return type; }

private:
  BaseType *type;
  Location loc;
};

} // namespace rust_compiler::sema::type_checking::TyTy
