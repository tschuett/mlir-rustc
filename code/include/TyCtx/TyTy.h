#pragma once

#include "AST/Expression.h"
#include "AST/GenericArgs.h"
#include "AST/GenericParams.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Trait.h"
#include "Basic/Ids.h"
#include "Basic/Mutability.h"
#include "Bounds.h"
#include "Lexer/Identifier.h"
#include "Location.h"
#include "Substitutions.h"
#include "TyCtx/ItemIdentity.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/Predicate.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TraitReference.h"
#include "TypeIdentity.h"

#include <optional>
#include <set>
#include <vector>

// FIXME
namespace rust_compiler::sema::type_checking {
class TypeResolver;
}

namespace rust_compiler::tyctx::TyTy {

using namespace rust_compiler::adt;
using namespace rust_compiler::basic;

enum class FunctionTrait { FnOnce, Fn, FnMut };

class BaseType;

class Argument {
public:
  Argument(NodeIdentity ident, TyTy::BaseType *argumentType, Location loc)
      : ident(ident), argumentType(argumentType), loc(loc) {}

  TyTy::BaseType *getArgumentType() const { return argumentType; }

private:
  NodeIdentity ident;
  TyTy::BaseType *argumentType;
  Location loc;
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
  /// Placeholder type in traits
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
  void setTypeReference(basic::NodeId id) { typeReference = id; }

  std::set<basic::NodeId> getCombinedReferences() const { return combined; }
  void appendReference(basic::NodeId id);

  TypeKind getKind() const { return kind; }

  bool hasSubsititionsDefined() const;
  bool needsGenericSubstitutions() const;
  virtual std::string toString() const = 0;

  virtual unsigned getNumberOfSpecifiedBounds() const = 0;

  bool satisfiesBound(const TypeBoundPredicate &predicate) const;
  bool isBoundsCompatible(const BaseType &other, Location loc,
                          bool emitError) const;

  bool isUnit() const;

  bool isConcrete() const;

  const BaseType *destructure() const;
  BaseType *destructure();

  virtual BaseType *clone() const = 0;

  void inheritBounds(const BaseType &other);
  void inheritBounds(std::vector<TyTy::TypeBoundPredicate> specifiedBounds);

  TypeIdentity getTypeIdentity() const { return identity; }

  Location getLocation() const { return identity.getLocation(); }

  virtual bool canEqual(const BaseType *other, bool emitErrors) const = 0;

  bool boundsCompatible(const BaseType *other, Location loc,
                        bool emitError) const;

  // FIXME this will eventually go away
  const BaseType *getRoot() const;

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
  unsigned getNumberOfSpecifiedBounds() const override;

  IntKind getIntKind() const;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

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
  unsigned getNumberOfSpecifiedBounds() const override;

  UintKind getUintKind() const { return kind; }

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

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
  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
};

class ISizeType : public BaseType {
public:
  ISizeType(basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  ISizeType(basic::NodeId, basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
};

/// F32 or F64
class FloatType : public BaseType {
public:
  FloatType(basic::NodeId, FloatKind,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  FloatType(basic::NodeId ref, basic::NodeId type, FloatKind,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() const override;

  FloatKind getFloatKind() const { return kind; }

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

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

  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
};

class CharType : public BaseType {
public:
  CharType(basic::NodeId,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  CharType(basic::NodeId, basic::NodeId,
           std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
};

class StrType : public BaseType {
public:
  StrType(basic::NodeId,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  StrType(basic::NodeId, basic::NodeId,
          std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
};

/// !
class NeverType : public BaseType {
public:
  NeverType(basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());
  NeverType(basic::NodeId, basic::NodeId,
            std::set<basic::NodeId> refs = std::set<basic::NodeId>());

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
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
  unsigned getNumberOfSpecifiedBounds() const override;

  static TupleType *getUnitType(basic::NodeId);

  size_t getNumberOfFields() const { return fields.size(); }
  BaseType *getField(size_t i) const { return fields[i].getType(); }

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

private:
  std::vector<TypeVariable> fields;
};

class FunctionType : public BaseType, public SubstitutionRef {
public:
  static constexpr uint8_t FunctionTypeDefaultFlags = 0x00;
  static constexpr uint8_t FunctionTypeIsMethod = 0x01;
  static constexpr uint8_t FunctionTypeIsExtern = 0x02;
  static constexpr uint8_t FunctionTypeIsVariadic = 0x04;

  FunctionType(
      basic::NodeId id, lexer::Identifier name, tyctx::TypeIdentity ident,
      uint8_t flags,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::vector<SubstitutionParamMapping> substRefs,
      std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, id, TypeKind::Function, ident, refs),
        SubstitutionRef(substRefs, SubstitutionArgumentMappings::error()),
        name(name), parameters(parameters), returnType(returnType) {}

  FunctionType(
      basic::NodeId id, basic::NodeId typeId, lexer::Identifier name,
      tyctx::TypeIdentity ident, uint8_t flags,
      std::vector<std::pair<
          std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
          TyTy::BaseType *>>
          parameters,
      TyTy::BaseType *returnType,
      std::vector<SubstitutionParamMapping> substRefs,
      std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, typeId, TypeKind::Function, ident, refs),
        SubstitutionRef(substRefs, SubstitutionArgumentMappings::error()),
        name(name), parameters(parameters), returnType(returnType) {}

  std::string toString() const override;

  TyTy::BaseType *getReturnType() const;

  unsigned getNumberOfSpecifiedBounds() const override;

  // basic::NodeId getId() const { return id; }

  Identifier getIdentifier() const { return name; }

  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                TyTy::BaseType *>>
  getParameters() const {
    return parameters;
  }

  BaseType *clone() const final override;

  bool isMethod() const {
    if (parameters.size() == 0)
      return false;
    return (flags & FunctionTypeIsMethod) == FunctionTypeIsMethod;
  }

  bool isVaradic() const {
    return (flags & FunctionTypeIsVariadic) == FunctionTypeIsVariadic;
  }

  size_t getNumberOfArguments() const { return parameters.size(); }

  std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
            TyTy::BaseType *>
  getParameter(size_t i) const {
    return parameters[i];
  }

  FunctionType *
  handleSubstitions(SubstitutionArgumentMappings &mappings) override final;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

private:
  lexer::Identifier name;
  uint8_t flags;
  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                TyTy::BaseType *>>
      parameters;
  TyTy::BaseType *returnType;
};

/// ClosureSubsts
/// https://doc.rust-lang.org/stable/nightly-rustc/rustc_middle/ty/struct.ClosureSubsts.html
class ClosureType : public BaseType, public SubstitutionRef {
public:
  ClosureType(basic::NodeId id, TypeIdentity ident, TupleType *closureArgs,
              TypeVariable resultType,
              std::vector<SubstitutionParamMapping> substRefs,
              std::set<basic::NodeId> captures,
              std::set<basic::NodeId> refs = std::set<basic::NodeId>(),
              std::vector<TypeBoundPredicate> specifiedBounds =
                  std::vector<TypeBoundPredicate>())
      : BaseType(id, id, TypeKind::Closure, ident, refs),
        SubstitutionRef(substRefs, SubstitutionArgumentMappings::error()),
        parameters(closureArgs), resultType(resultType), captures(captures) {
    inheritBounds(specifiedBounds);
  };

  ClosureType(basic::NodeId id, basic::NodeId type, TypeIdentity ident,
              TupleType *closureArgs, TypeVariable resultType,
              std::vector<SubstitutionParamMapping> substRefs,
              std::set<basic::NodeId> captures,
              std::set<basic::NodeId> refs = std::set<basic::NodeId>(),
              std::vector<TypeBoundPredicate> specifiedBounds =
                  std::vector<TypeBoundPredicate>())
      : BaseType(id, type, TypeKind::Closure, ident, refs),
        SubstitutionRef(substRefs, SubstitutionArgumentMappings::error()),
        parameters(closureArgs), resultType(resultType), captures(captures) {
    inheritBounds(specifiedBounds);
  };

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() const override;

  TyTy::TupleType *getParameters() const { return parameters; }
  TyTy::BaseType *getResultType() const { return resultType.getType(); }

  BaseType *clone() const final override;

  void setupFnOnceOutput() const;

  ClosureType *
  handleSubstitions(SubstitutionArgumentMappings &mappings) override final;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

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
  unsigned getNumberOfSpecifiedBounds() const override;

  void applyScalarTypeHint(const BaseType &hint);

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

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
  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
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

  Identifier getName() const { return identifier; }
  TyTy::BaseType *getFieldType() const { return type; }
  void setFieldType(TyTy::BaseType *fieldType) { type = fieldType; }

  //  std::string toString() const override;
  //  unsigned getNumberOfSpecifiedBounds() override;

  StructFieldType *clone() const;

  std::string toString() const;

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
  Identifier getIdentifier() const { return identifier; }

  std::vector<TyTy::StructFieldType *> getFields() const { return fields; }
  size_t getNumberOfFields() const { return fields.size(); }

  TyTy::StructFieldType *getFieldAt(size_t i) const { return fields[i]; }

  VariantDef *clone() const;

  bool lookupField(lexer::Identifier, TyTy::StructFieldType **lookup,
                   size_t *index);

  bool isDatalessVariant() const { return kind == VariantKind::Enum; }
  static VariantDef &getErrorNode() {
    static VariantDef node = {basic::UNKNOWN_NODEID, lexer::Identifier(""),
                              TypeIdentity::empty(), nullptr};
    return node;
  }

  std::string toString() const;

  std::string VariantKind2String() const;

private:
  basic::NodeId id;
  adt::Identifier identifier;
  TypeIdentity ident;
  VariantKind kind;
  ast::Expression *discriminant;
  std::vector<TyTy::StructFieldType *> fields;
};

enum class ADTKind { StructStruct, TupleStruct, Enum };

class ADTType : public BaseType, public SubstitutionRef {
public:
  ADTType(basic::NodeId id, const adt::Identifier &identifier,
          TypeIdentity ident, ADTKind kind, std::vector<VariantDef *> variants,
          std::vector<SubstitutionParamMapping> substRefs,
          SubstitutionArgumentMappings genericArguments =
              SubstitutionArgumentMappings::error(),
          std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, id, TypeKind::ADT, ident, refs),
        SubstitutionRef(substRefs, genericArguments), identifier(identifier),
        variants(variants), kind(kind) {}

  ADTType(basic::NodeId id, basic::NodeId type,
          const adt::Identifier &identifier, TypeIdentity ident, ADTKind kind,
          std::vector<VariantDef *> variants,
          std::vector<SubstitutionParamMapping> substRefs,
          SubstitutionArgumentMappings genericArguments =
              SubstitutionArgumentMappings::error(),
          std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, type, TypeKind::ADT, ident, refs),
        SubstitutionRef(substRefs, genericArguments), identifier(identifier),
        variants(variants), kind(kind) {}

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() const override;

  ADTKind getKind() const { return kind; }

  bool isEnum() const { return kind == ADTKind::Enum; };

  std::vector<VariantDef *> getVariants() const { return variants; }
  VariantDef *getVariant(size_t i) const { return variants[i]; }

  BaseType *clone() const final override;

  size_t getNumberOfVariants() const { return variants.size(); }

  bool lookupVariant(const Identifier &ident, VariantDef **found) const {
    for (VariantDef *variant : variants) {
      if (variant->getIdentifier() == ident) {
        *found = variant;
        return true;
      }
    }
    return false;
  }

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

  ADTType *
  handleSubstitions(SubstitutionArgumentMappings &mappings) override final;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

  Identifier getIdentifier() const { return identifier; }

private:
  adt::Identifier identifier;
  std::vector<VariantDef *> variants;
  ADTKind kind;
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

  Identifier getSymbol() const { return identifier; }

  unsigned getNumberOfSpecifiedBounds() const override;

  bool canResolve() const { return getReference() == getTypeReference(); }
  BaseType *resolve() const;

  BaseType *clone() const final override;

  bool isImplicitSelfTrait() const;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

private:
  Identifier identifier;
  Location loc;
  ast::TypeParam type;
  std::vector<TyTy::TypeBoundPredicate> bounds;
  bool isTraitSelf;
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
  unsigned getNumberOfSpecifiedBounds() const override;
  TyTy::BaseType *getElementType() const;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

  std::shared_ptr<ast::Expression> getCapacityExpression() const {
    return expr;
  }

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

  std::string toString() const override;
  unsigned getNumberOfSpecifiedBounds() const override;

  bool isMutable() const { return mut == Mutability::Mut; }

  bool canEqual(const BaseType *other, bool emitErrors) const override;

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

  TyTy::BaseType *getReturnType() const { return resultType.getType(); }
  std::vector<TypeVariable> getParameters() const { return parameters; }

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

  size_t getNumberOfArguments() const { return parameters.size(); }

  BaseType *getParameter(size_t i) const { return parameters[i].getType(); }

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

  bool canEqual(const BaseType *other, bool emitErrors) const override;

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() const override { return 0; }

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

  std::optional<TyTy::StrType *> isDynStrType() const;

  bool isMutable() const { return mut == Mutability::Mut; }

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() const override { return 0; }

  bool canEqual(const BaseType *other, bool emitErrors) const override;

  Mutability getMut() const { return mut; }

private:
  TypeVariable base;
  Mutability mut;
};

/// https://doc.rust-lang.org/book/ch19-03-advanced-traits.html
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

  void setAssociatedType(basic::NodeId);

  bool canEqual(const BaseType *other, bool emitErrors) const override;

private:
  Identifier id;
};

class ProjectionType : public BaseType, public SubstitutionRef {
public:
  const BaseType *get() const { return base; }
  BaseType *get() { return base; }

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;

private:
  BaseType *base;
};

class DynamicObjectType : public BaseType {
public:
  DynamicObjectType(basic::NodeId id, TypeIdentity ident,
                    std::vector<TypeBoundPredicate> specifiedBounds,
                    std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, id, TypeKind::Dynamic, ident, specifiedBounds, refs) {}
  DynamicObjectType(basic::NodeId id, basic::NodeId type, TypeIdentity ident,
                    std::vector<TypeBoundPredicate> specifiedBounds,
                    std::set<basic::NodeId> refs = std::set<basic::NodeId>())
      : BaseType(id, type, TypeKind::Dynamic, ident, specifiedBounds, refs) {}

  std::string toString() const override;

  unsigned getNumberOfSpecifiedBounds() const override;

  BaseType *clone() const final override;

  bool canEqual(const BaseType *other, bool emitErrors) const override;
};

std::string TypeKind2String(TypeKind kind);

} // namespace rust_compiler::tyctx::TyTy
