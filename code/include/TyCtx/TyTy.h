#pragma once

#include "AST/Expression.h"
#include "AST/GenericArgs.h"
#include "AST/GenericParams.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Location.h"
// #include "Substitutions.h"
#include "TyCtx/ItemIdentity.h"
#include "TyCtx/NodeIdentity.h"
#include "TypeIdentity.h"

#include <set>

namespace rust_compiler::tyctx::TyTy {

using namespace rust_compiler::adt;

enum class Mutability { Imm, Mut };

class TypeBoundPredicate {
public:
  bool isError() const { return error; }

  basic::NodeId getId() const { return id; }

private:
  basic::NodeId id;
  bool error = false;
};

class TypeBoundsMappings {
public:
  std::vector<TypeBoundPredicate> &getSpecifiedBounds() {
    return specifiedBounds;
  }

  const std::vector<TypeBoundPredicate> &getSpecifiedBounds() const {
    return specifiedBounds;
  }

protected:
  TypeBoundsMappings(std::vector<TypeBoundPredicate> specifiedBounds)
      : specifiedBounds(specifiedBounds) {}

  void addBound(const TypeBoundPredicate &predicate);

private:
  std::vector<TypeBoundPredicate> specifiedBounds;
};

/// https://rustc-dev-guide.rust-lang.org/generics.html
class GenericParameters {
public:
  GenericParameters(std::optional<ast::GenericParams> genericParams)
      : genericParams(genericParams) {}

  bool needsSubstitution() const;

  std::optional<ast::GenericParams> getGenericParams() const {
    return genericParams;
  }

protected:
  std::string substToString() const;

private:
  std::optional<ast::GenericParams> genericParams;
};

/// https://rustc-dev-guide.rust-lang.org/ty.html
/// https://doc.rust-lang.org/reference/types.html
/// https://rustc-dev-guide.rust-lang.org/type-inference.html#inference-variables
/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/sty/enum.TyKind.html
/// https://rustc-dev-guide.rust-lang.org/ty.html?highlight=adt#adts-representation

enum class InferKind { Integral, Float, General };

/// Defines the kinds of types used by the type system.
enum class TypeKind {
  /// The primitive boolean type. Written as bool.
  Bool,
  /// The primitive character type; holds a Unicode scalar value (a
  /// non-surrogate code point). Written as char.
  Char,
  /// A primitive signed integer type.
  Int,
  /// A primitive unsigned integer type.
  Uint,
  USize,
  ISize,
  /// A primitive floating-point type.
  Float,
  /// The anonymous type of closure.
  Closure,
  /// The anonymous type of a function declaration/definition. Each
  /// function has a unique type.
  Function,
  /// A type variable used during type checking.
  Inferred,
  /// The never type !.
  Never,
  /// The pointee of a string slice. Written as str.
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

  TypeVariable clone() const;

private:
  basic::NodeId id;
};

class BaseType : public TypeBoundsMappings {
public:
  virtual ~BaseType() = default;

  basic::NodeId getReference() const;
  basic::NodeId getTypeReference() const;

  void setReference(basic::NodeId);

  std::set<basic::NodeId> getCombinedReferences() const { return combined; }
  void appendReference(basic::NodeId id);

  TypeKind getKind() const { return kind; }

  bool needsGenericSubstitutions() const;
  virtual std::string toString() const = 0;

  virtual unsigned getNumberOfSpecifiedBounds() = 0;

  bool isConcrete() const;

  const BaseType *destructure() const;
  BaseType *destructure();

  virtual BaseType *clone() const = 0;

  void inheritBounds(std::span<TyTy::TypeBoundPredicate> specifiedBounds);

  TypeIdentity getTypeIdentity() const { return identity; }

protected:
  BaseType(basic::NodeId ref, basic::NodeId typeRef, TypeKind kind,
           TypeIdentity ident,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  BaseType(basic::NodeId ref, basic::NodeId typeRef, TypeKind kind,
           TypeIdentity ident, std::vector<TypeBoundPredicate> specifiedBounds,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());

private:
  basic::NodeId reference;
  basic::NodeId typeReference;
  TypeKind kind;
  TypeIdentity identity;
  std::set<basic::NodeId> combined;
};

class IntType : public BaseType {
public:
  IntType(basic::NodeId, IntKind,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  IntType(basic::NodeId, basic::NodeId, IntKind,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  IntKind getIntKind() const;

  BaseType *clone() const final override;

private:
  IntKind kind;
};

class UintType : public BaseType {
public:
  UintType(basic::NodeId, UintKind,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  UintType(basic::NodeId, basic::NodeId, UintKind,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  UintKind getUintKind() const { return kind; }

  BaseType *clone() const final override;

private:
  UintKind kind;
};

class USizeType : public BaseType {
public:
  USizeType(basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  USizeType(basic::NodeId, basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
};

class ISizeType : public BaseType {
public:
  ISizeType(basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  ISizeType(basic::NodeId, basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
};

/// F32 or F64
class FloatType : public BaseType {
public:
  FloatType(basic::NodeId, FloatKind,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  FloatType(basic::NodeId ref, basic::NodeId type, FloatKind,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  FloatKind getFloatKind() const { return kind; }

  BaseType *clone() const final override;

private:
  FloatKind kind;
};

/// false or true
class BoolType : public BaseType {
public:
  BoolType(basic::NodeId,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  BoolType(basic::NodeId, basic::NodeId,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
};

class CharType : public BaseType {
public:
  CharType(basic::NodeId,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  CharType(basic::NodeId, basic::NodeId,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
};

class StrType : public BaseType {
public:
  StrType(basic::NodeId,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  StrType(basic::NodeId, basic::NodeId,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
};

/// !
class NeverType : public BaseType {
public:
  NeverType(basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  NeverType(basic::NodeId, basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
};

/// (.., .., ..)
class TupleType : public BaseType {
public:
  TupleType(basic::NodeId, Location loc,
            std::vector<TyTy::TypeVariable> parameterTypes =
                std::vector<TyTy::TypeVariable>(),
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  TupleType(basic::NodeId, basic::NodeId, Location loc,
            std::vector<TyTy::TypeVariable> parameterTypes =
                std::vector<TyTy::TypeVariable>(),
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  static TupleType *getUnitType(basic::NodeId);

  size_t getNumberOfFields() const { return fields.size(); }
  BaseType *getField(size_t i) const { return fields[i].getType(); }

  BaseType *clone() const final override;

private:
  std::vector<TypeVariable> fields;
};

class FunctionType : public BaseType, public GenericParameters {
public:
  FunctionType(
      basic::NodeId, lexer::Identifier name, tyctx::ItemIdentity,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::optional<ast::GenericParams> genericParams,
      std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  FunctionType(
      basic::NodeId, basic::NodeId, lexer::Identifier name, tyctx::ItemIdentity,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::optional<ast::GenericParams> genericParams,
      std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  TyTy::BaseType *getReturnType() const;

  unsigned getNumberOfSpecifiedBounds() override;

  basic::NodeId getId() const { return id; }

  Identifier getIdentifier() const { return name; }

  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                TyTy::BaseType *>>
  getParameters() const;

  BaseType *clone() const final override;

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
class ClosureType : public BaseType, public GenericParameters {
public:
  ClosureType(basic::NodeId id, TypeIdentity ident, TupleType *closureArgs,
              TypeVariable resultType,
              std::optional<ast::GenericParams> genericParams,
              std::set<basic::NodeId> captures,
              std::set<basic::NodeId> refs = std::set<basic::NodeId>(),
              std::vector<TypeBoundPredicate> specifiedBounds =
                  std::vector<TypeBoundPredicate>())
      : BaseType(id, id, TypeKind::Closure, ident),
        GenericParameters(genericParams), parameters(closureArgs),
        resultType(resultType), captures(captures) {
    inheritBounds(specifiedBounds);
  };

  ClosureType(basic::NodeId id, basic::NodeId type, TypeIdentity ident,
              TupleType *closureArgs, TypeVariable resultType,
              std::optional<ast::GenericParams> genericParams,
              std::set<basic::NodeId> captures,
              std::set<basic::NodeId> refs = std::set<basic::NodeId>(),
              std::vector<TypeBoundPredicate> specifiedBounds =
                  std::vector<TypeBoundPredicate>())
      : BaseType(id, type, TypeKind::Closure, ident),
        GenericParameters(genericParams), parameters(closureArgs),
        resultType(resultType), captures(captures) {
    inheritBounds(specifiedBounds);
  };

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  TyTy::TupleType *getParameters() const { return parameters; }
  TyTy::BaseType *getResultType() const { return resultType.getType(); }

  BaseType *clone() const final override;

private:
  TyTy::TupleType *parameters;
  TyTy::TypeVariable resultType;
  std::set<basic::NodeId> captures;
};

class InferType : public BaseType {
public:
  InferType(basic::NodeId, InferKind kind, TypeHint hint, Location loc,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  InferType(basic::NodeId, basic::NodeId, InferKind kind, TypeHint hint,
            Location loc,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  InferKind getInferredKind() const { return inferKind; }

  bool needsSubstitution() const;
  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  void applyScalarTypeHint(const BaseType &hint);

  BaseType *clone() const final override;

private:
  InferKind inferKind;
  TypeHint defaultHint;
  Location loc;
};

class ErrorType : public BaseType {
public:
  ErrorType(basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  ErrorType(basic::NodeId, basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  BaseType *clone() const final override;
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
  //  StructFieldType(basic::NodeId, std::string_view, TyTy::BaseType *,
  //                  Location loc);

  TyTy::BaseType *getFieldType() const { return type; }

  //  std::string toString() const override;
  //  unsigned getNumberOfSpecifiedBounds() override;

  StructFieldType *clone() const;

private:
  basic::NodeId ref;
  TyTy::BaseType *type;
  Location loc;
  adt::Identifier identifier;
};

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.VariantDef.html
enum class VariantKind { Enum, Struct, Tuple };

class VariantDef {
public:
  VariantDef(basic::NodeId, const adt::Identifier &, TypeIdentity,
             ast::Expression *discriminant);
  VariantDef(basic::NodeId, const adt::Identifier &, TypeIdentity, VariantKind,
             ast::Expression *discriminant,
             std::vector<TyTy::StructFieldType *>);

  VariantKind getKind() const { return kind; }
  basic::NodeId getId() const { return id; }

  std::vector<TyTy::StructFieldType *> getFields() const;

  VariantDef *clone() const;

  static VariantDef &getErrorNode() {
    static VariantDef node = {basic::UNKNOWN_NODEID, lexer::Identifier(""),
                              TypeIdentity::empty(), nullptr};
    return node;
  }

private:
  basic::NodeId id;
  adt::Identifier identifier;
  TypeIdentity ident;
  VariantKind kind;
  ast::Expression *discriminant;
  std::vector<TyTy::StructFieldType *> fields;
};

enum class ADTKind { StructStruct, TupleStruct, Enum };

class ADTType : public BaseType, public GenericParameters {
public:
  ADTType(basic::NodeId, const adt::Identifier &, TypeIdentity, ADTKind,
          std::span<VariantDef *>, std::optional<ast::GenericParams>,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  ADTType(basic::NodeId, basic::NodeId, const adt::Identifier &, TypeIdentity,
          ADTKind, std::span<VariantDef *>, std::optional<ast::GenericParams>,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;

  ADTKind getKind() const { return kind; }

  bool isUnit() const;

  bool isEnum() const { return kind == ADTKind::Enum; };

  std::vector<VariantDef *> getVariants() const { return variants; }

  BaseType *clone() const final override;

  size_t getNumberOfVariants() const { return variants.size(); }

  bool lookupVariantById(basic::NodeId id, VariantDef **found,
                         int *index = nullptr) {
    int i = 0;
    for (VariantDef *variant : variants) {
      if (variant->getId() == id) {
        if (index != nullptr)
          *index = i;
        *found = variant;
        return true;
      }
      ++i;
    }
    return false;
  }

private:
  adt::Identifier identifier;
  ADTKind kind;
  std::vector<VariantDef *> variants;
};

class ParamType : public BaseType {
public:
  ParamType(const Identifier &identifier, Location loc, basic::NodeId id,
            const ast::TypeParam &tpe,
            std::vector<TyTy::TypeBoundPredicate> preds,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, id, TypeKind::Parameter, TypeIdentity::empty(), refs),
        identifier(identifier), loc(loc), type(tpe), bounds(preds) {}
  ParamType(const Identifier &identifier, Location loc, basic::NodeId id,
            basic::NodeId typeId, const ast::TypeParam &tpe,
            std::vector<TyTy::TypeBoundPredicate> preds,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, typeId, TypeKind::Parameter, TypeIdentity::empty(), refs),
        identifier(identifier), loc(loc), type(tpe), bounds(preds) {}

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() override;

  bool canResolve() const { return getReference() == getTypeReference(); }
  BaseType *resolve() const;

  BaseType *clone() const final override;

private:
  Identifier identifier;
  Location loc;
  ast::TypeParam type;
  std::vector<TyTy::TypeBoundPredicate> bounds;
};

class ArrayType : public BaseType {
public:
  ArrayType(basic::NodeId id, Location loc,
            std::shared_ptr<ast::Expression> expr, TypeVariable type,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, id, TypeKind::Array, TypeIdentity::empty(), refs),
        loc(loc), expr(expr), type(type) {}
  ArrayType(basic::NodeId id, basic::NodeId typeId, Location loc,
            std::shared_ptr<ast::Expression> expr, TypeVariable type,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, typeId, TypeKind::Array, TypeIdentity::empty(), refs),
        loc(loc), expr(expr), type(type) {}

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() override;
  TyTy::BaseType *getElementType() const;

  BaseType *clone() const final override;

private:
  Location loc;
  std::shared_ptr<ast::Expression> expr;
  TypeVariable type;
};

class RawPointerType : public BaseType {
public:
  RawPointerType(basic::NodeId id, TypeVariable type, Mutability mut,
                 std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, id, TypeKind::RawPointer, TypeIdentity::empty(), refs),
        base(type), mut(mut){};

  RawPointerType(basic::NodeId id, basic::NodeId typeId, TypeVariable type,
                 Mutability mut,
                 std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, typeId, TypeKind::RawPointer, TypeIdentity::empty(), refs),
        base(type), mut(mut){};

  BaseType *getBase() const { return base.getType(); }

  BaseType *clone() const final override;

private:
  TypeVariable base;
  Mutability mut;
};

class FunctionPointerType : public BaseType {
public:
  FunctionPointerType(basic::NodeId ref, Location loc,
                      std::vector<TypeVariable> params, TypeVariable resultType,
                      std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, ref, TypeKind::FunctionPointer, TypeIdentity::empty(),
                 refs),
        parameters(params), resultType(resultType) {}
  FunctionPointerType(basic::NodeId ref, basic::NodeId typeRef, Location loc,
                      std::vector<TypeVariable> params, TypeVariable resultType,
                      std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, typeRef, TypeKind::FunctionPointer, TypeIdentity::empty(),
                 refs),
        parameters(params), resultType(resultType) {}

  TyTy::BaseType *getReturnType() const;
  std::vector<TypeVariable> getParameters() const;

  BaseType *clone() const final override;

private:
  std::vector<TypeVariable> parameters;
  TypeVariable resultType;
};

class SliceType : public BaseType {
public:
  SliceType(basic::NodeId ref, Location loc, TypeVariable base,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, ref, TypeKind::Slice, TypeIdentity::empty(), refs),
        elementType(base) {}
  SliceType(basic::NodeId ref, basic::NodeId typeRef, Location loc,
            TypeVariable base,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, typeRef, TypeKind::Slice, TypeIdentity::empty(), refs),
        elementType(base) {}

  TyTy::BaseType *getElementType() const { return elementType.getType(); }

  BaseType *clone() const final override;

private:
  TypeVariable elementType;
};

class ReferenceType : public BaseType {
public:
  ReferenceType(basic::NodeId ref, TypeVariable base, Mutability mut,
                std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, ref, TypeKind::Reference, TypeIdentity::empty(), refs),
        base(base), mut(mut) {}
  ReferenceType(basic::NodeId ref, basic::NodeId typeRef, TypeVariable base,
                Mutability mut,
                std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, typeRef, TypeKind::Reference, TypeIdentity::empty(),
                 refs),
        base(base), mut(mut) {}

  BaseType *getBase() const { return base.getType(); }

  BaseType *clone() const final override;

private:
  TypeVariable base;
  Mutability mut;
};

class PlaceholderType : public BaseType {
public:
  PlaceholderType(const Identifier &id, basic::NodeId ref,
                  std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, ref, TypeKind::PlaceHolder, TypeIdentity::empty(), refs),
        id(id) {}
  PlaceholderType(const Identifier &id, basic::NodeId ref,
                  basic::NodeId typeRef,
                  std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(ref, typeRef, TypeKind::PlaceHolder, TypeIdentity::empty(),
                 refs),
        id(id) {}

  bool canResolve() const;

  BaseType *resolve() const;

  BaseType *clone() const final override;

private:
  Identifier id;
};

class ProjectionType : public BaseType {
public:
  const BaseType *get() const { return base; }
  BaseType *get() { return base; }

  BaseType *clone() const final override;

private:
  BaseType *base;
};

class DynamicObjectType : public BaseType {};

} // namespace rust_compiler::tyctx::TyTy
