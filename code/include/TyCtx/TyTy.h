#pragma once

#include "AST/Expression.h"
#include "AST/GenericArgs.h"
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

using namespace rust_compiler::adt;

/// https://doc.rust-lang.org/reference/types.html
/// https://rustc-dev-guide.rust-lang.org/type-inference.html#inference-variables
/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html

enum class InferKind { Integral, Float, General };

/// Defines the kinds of types used by the type system.
enum class TypeKind {
  Bool,
  Char,
  Int,
  Uint,
  USize,
  ISize,
  Float,
  /// The anonymous type of closure.
  Closure,
  Function,
  Inferred,
  /// The never type !.
  Never,
  Str,
  /// A tuple type.
  Tuple,
  /// A type parameter.
  Parameter,
  /// Algebraic data types: struct, enum, and union
  ADT,
  /// An array with a given length.
  Array,
  /// The pointee of an array slice. Written as [T].
  Slice,
  Projection,
  /// A trait object
  Dynamic,
  PlaceHolder,
  /// A pointer to a function
  FunctionPointer,
  /// A raw pointer. Written as *mut T or *const T.
  RawPointer,
  /// A reference. A pointer with an associated lifetime. Written as &'a mut T
  /// or &'a T.
  Reference,

  Error
};

enum class IntKind { I8, I16, I32, I64, I128 };

enum class UintKind { U8, U16, U32, U64, U128 };

enum class FloatKind { F32, F64 };

enum class SignedHint { Signed, Unsigned, Unkown };
enum class SizeHint { S8, S16, S32, S64, S128, Unknown };

class TypeHint {
public:
  TypeKind kind;
  SignedHint signHint;
  SizeHint sizeHint;

  SignedHint getSignedHint() const { return signHint; }
  SizeHint getSizeHint() const { return sizeHint; }

  static TypeHint unknown() {
    return TypeHint{TypeKind::Error, SignedHint::Unkown, SizeHint::Unknown};
  }
};

bool isIntegerLike(TypeKind);
bool isSignedIntegerLike(TypeKind);
bool isFloatLike(TypeKind);

class BaseType;

class TypeVariable {
public:
  TypeVariable(basic::NodeId id);
  TyTy::BaseType *getType() const;

  static TypeVariable getImplicitInferVariable(Location);

  bool isConcrete() const;

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

  bool needsGenericSubstitutions() const;
  virtual std::string toString() const = 0;

  virtual unsigned getNumberOfSpecifiedBounds() = 0;

  bool isConcrete() const;

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

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  IntKind getIntKind() const;

private:
  IntKind kind;
};

class UintType : public BaseType {
public:
  UintType(basic::NodeId, UintKind);

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  UintKind getUintKind() const { return kind; }

private:
  UintKind kind;
};

class USizeType : public BaseType {
public:
  USizeType(basic::NodeId);

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
};

class ISizeType : public BaseType {
public:
  ISizeType(basic::NodeId);

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
};

/// F32 or F64
class FloatType : public BaseType {
public:
  FloatType(basic::NodeId, FloatKind);

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  FloatKind getFloatKind() const { return kind; }

private:
  FloatKind kind;
};

/// false or true
class BoolType : public BaseType {
public:
  BoolType(basic::NodeId);

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

class CharType : public BaseType {
public:
  CharType(basic::NodeId);

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

class StrType : public BaseType {
public:
  StrType(basic::NodeId);

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

/// !
class NeverType : public BaseType {
public:
  NeverType(basic::NodeId);

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;
};

/// (.., .., ..)
class TupleType : public BaseType {
public:
  TupleType(basic::NodeId, Location loc,
            std::span<TyTy::TypeVariable> parameterTypes);
  TupleType(basic::NodeId, Location loc);

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  static TupleType *getUnitType(basic::NodeId);

  size_t getNumberOfFields() const { return fields.size(); }
  BaseType *getField(size_t i) const { return fields[i].getType(); }

private:
  std::vector<TypeVariable> fields;
};

class FunctionType : public BaseType, public SubstitutionsReference {
public:
  FunctionType(
      basic::NodeId, lexer::Identifier name, tyctx::ItemIdentity,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::vector<TyTy::SubstitutionParamMapping> substitutions,
      std::optional<ast::GenericParams> genericParams);

  std::string toString() const override;

  TyTy::BaseType *getReturnType() const;

  unsigned getNumberOfSpecifiedBounds() override;

  basic::NodeId getId() const { return id; }

  Identifier getIdentifier() const { return name; }

  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                TyTy::BaseType *>>
  getParameters() const;

private:
  basic::NodeId id;
  lexer::Identifier name;
  tyctx::ItemIdentity ident;
  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                TyTy::BaseType *>>
      parameters;
  TyTy::BaseType *returnType = nullptr;
};

/// ClosureSubsts
/// https://doc.rust-lang.org/stable/nightly-rustc/rustc_middle/ty/struct.ClosureSubsts.html
class ClosureType : public BaseType, public SubstitutionsReference {
public:
  ClosureType(basic::NodeId id, Location loc, TypeIdentity ident,
              TupleType *closureArgs, TypeVariable resultType,
              std::vector<SubstitutionParamMapping> substitutions,
              std::optional<ast::GenericParams> genericParams,
              std::set<basic::NodeId> captures)
      : BaseType(id, id, TypeKind::Closure, ident),
        SubstitutionsReference(substitutions, genericParams),
        parameters(closureArgs), resultType(resultType), captures(captures){};

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  TyTy::TupleType *getParameters() const { return parameters; }
  TyTy::BaseType *getResultType() const { return resultType.getType(); }

private:
  TyTy::TupleType *parameters;
  TyTy::TypeVariable resultType;
  std::set<basic::NodeId> captures;
};

class InferType : public BaseType {
public:
  InferType(basic::NodeId, InferKind kind, TypeHint hint, Location loc);

  InferKind getInferredKind() const { return inferKind; }

  bool needsSubstitution() const;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  void applyScalarTypeHint(const BaseType &hint);

private:
  InferKind inferKind;
  TypeHint defaultHint;
};

class ErrorType : public BaseType {
public:
  ErrorType(basic::NodeId);

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

class StructFieldType {
public:
  StructFieldType(basic::NodeId, const adt::Identifier &, TyTy::BaseType *,
                  Location loc);
  StructFieldType(basic::NodeId, std::string_view, TyTy::BaseType *,
                  Location loc);

  TyTy::BaseType *getFieldType() const { return type; }

  //  std::string toString() const override;
  //  unsigned getNumberOfSpecifiedBounds() override;

private:
  [[maybe_unused]] basic::NodeId ref;
  TyTy::BaseType *type;
  Location loc;
  std::optional<adt::Identifier> id;
  std::optional<std::string> idStr;
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.VariantDef.html
enum class VariantKind { Enum, Struct, Tuple };

class VariantDef {
public:
  VariantDef(basic::NodeId, const adt::Identifier &, TypeIdentity, VariantKind,
             std::span<TyTy::StructFieldType *>);

  VariantKind getKind() const { return kind; }
  basic::NodeId getId() const { return id; }

  std::vector<TyTy::StructFieldType *> getFields() const;

private:
  basic::NodeId id;
  adt::Identifier identifier;
  TypeIdentity ident;
  VariantKind kind;
  std::vector<TyTy::StructFieldType *> fields;
};

enum class ADTKind { StructStruct, TupleStruct };

class ADTType : public BaseType, public SubstitutionsReference {
public:
  ADTType(basic::NodeId, const adt::Identifier &, TypeIdentity, ADTKind,
          std::span<VariantDef *>, std::vector<SubstitutionParamMapping>,
          std::optional<ast::GenericParams>);

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  ADTKind getKind() const { return kind; }

  bool isUnit() const;

  std::vector<VariantDef *> getVariants() const;

private:
  adt::Identifier identifier;
  ADTKind kind;
  std::vector<VariantDef *> variants;
};

class TypeBoundPredicate {
public:
  bool isError() const { return error; }

private:
  bool error = false;
};

class ParamType : public BaseType {
public:
  ParamType(const Identifier &identifier, Location loc, basic::NodeId id,
            const ast::TypeParam &tpe,
            std::span<TyTy::TypeBoundPredicate> preds)
      : BaseType(id, id, TypeKind::Parameter, TypeIdentity::empty()),
        identifier(identifier), loc(loc), type(tpe) {
    bounds = {preds.begin(), preds.end()};
  }

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;

private:
  Identifier identifier;
  Location loc;
  ast::TypeParam type;
  std::vector<TyTy::TypeBoundPredicate> bounds;
};

class ArrayType : public BaseType {
public:
  ArrayType(basic::NodeId id, Location loc,
            std::shared_ptr<ast::Expression> expr, TypeVariable type)
      : BaseType(id, id, TypeKind::Array, TypeIdentity::empty()), loc(loc),
        expr(expr), type(type) {}

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
  TyTy::BaseType *getElementType() const;

private:
  Location loc;
  std::shared_ptr<ast::Expression> expr;
  TypeVariable type;
};

class RawPointerType : public BaseType {
public:
  BaseType *getBase() const { return base.getType(); }

private:
  TypeVariable base;
};

class FunctionPointerType : public BaseType {
public:
  TyTy::BaseType *getReturnType() const;
  std::vector<TypeVariable> getParameters() const;

private:
  std::vector<TypeVariable> parameters;
  TypeVariable resultType;
};

class SliceType : public BaseType {
public:
  TyTy::BaseType *getElementType() const { return elementType.getType(); }

private:
  TypeVariable elementType;
};

class ReferenceType : public BaseType {
public:
  BaseType *getBase() const { return base.getType(); }

private:
  TypeVariable base;
};

} // namespace rust_compiler::tyctx::TyTy
