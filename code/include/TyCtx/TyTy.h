#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Location.h"
#include "Substitutions.h"
#include "TyCtx/ItemIdentity.h"
#include "TyCtx/NodeIdentity.h"
#include "TypeIdentity.h"

#include <set>

namespace rust_compiler::tyctx::TyTy {

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

class BaseType {
public:
  virtual ~BaseType() = default;

  basic::NodeId getReference() const;
  basic::NodeId getTypeReference() const;

  void setReference(basic::NodeId);
  void appendReference(basic::NodeId);

  TypeKind getKind() const { return kind; }

  virtual bool needsGenericSubstitutions() const = 0;
  virtual std::string toString() const = 0;

  virtual unsigned getNumberOfSpecifiedBounds() = 0;

protected:
  BaseType(basic::NodeId ref, basic::NodeId ty_ref, TypeKind kind,
           TypeIdentity ident);

private:
  basic::NodeId reference;
  basic::NodeId typeReference;
  TypeKind kind;
  TypeIdentity identity;

  std::set<basic::NodeId> combined;
};

class IntType : public BaseType {
public:
  IntType(basic::NodeId, IntKind);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  IntKind getIntKind() const;

private:
  IntKind kind;
};

class UintType : public BaseType {
public:
  UintType(basic::NodeId, UintKind);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

private:
  UintKind kind;
};

class USizeType : public BaseType {
public:
  USizeType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
};

class ISizeType : public BaseType {
public:
  ISizeType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
};

class FloatType : public BaseType {
public:
  FloatType(basic::NodeId, FloatKind);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

private:
  FloatKind kind;
};

/// false or true
class BoolType : public BaseType {
public:
  BoolType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

class CharType : public BaseType {
public:
  CharType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

class StrType : public BaseType {
public:
  StrType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

/// !
class NeverType : public BaseType {
public:
  NeverType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

/// (.., .., ..)
class TupleType : public BaseType {
public:
  TupleType(basic::NodeId, Location loc);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  static TupleType *getUnitType(basic::NodeId);

private:
  std::vector<TypeVariable> fields;
};

class FunctionType : public BaseType {
public:
  FunctionType(
      basic::NodeId, lexer::Identifier name, tyctx::ItemIdentity,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::vector<TyTy::SubstitutionParamMapping> substitutions);

  std::string toString() const override;

  TyTy::BaseType *getReturnType() const;

  bool needsGenericSubstitutions() const override;

  unsigned getNumberOfSpecifiedBounds() override;

  basic::NodeId getId() const { return id; }

private:
  basic::NodeId id;
  lexer::Identifier name;
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
  InferType(basic::NodeId, InferKind kind, Location loc);

  InferKind getInferredKind() const { return inferKind; }

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

private:
  InferKind inferKind;
};

class ErrorType : public BaseType {
public:
  ErrorType(basic::NodeId);

  bool needsGenericSubstitutions() const override;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
};

class WithLocation {
public:
  WithLocation(BaseType *type, Location loc) : type(type), loc(loc) {}
  WithLocation(BaseType *type)
      : type(type), loc(Location::getEmptyLocation()) {}

  BaseType *getType() const { return type; }

  Location getLocation() const { return loc; }

private:
  BaseType *type;
  Location loc;
};

} // namespace rust_compiler::tyctx::TyTy
