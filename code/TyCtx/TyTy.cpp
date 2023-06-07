#include "TyCtx/TyTy.h"

#include "ADT/CanonicalPath.h"
#include "AST/Item.h"
#include "AST/Patterns/Patterns.h"
#include "AST/Trait.h"
#include "AST/VisItem.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Location.h"
#include "Session/Session.h"
#include "TyCtx/Bounds.h"
#include "TyCtx/Compare.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/SubstitutionsMapper.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TypeIdentity.h"

#include <functional>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <optional>
#include <sstream>
#include <vector>

using namespace rust_compiler::adt;
using namespace rust_compiler::tyctx;

namespace rust_compiler::tyctx::TyTy {

static constexpr uint32_t MAX_RECURSION_DEPTH = 1024 * 16;

bool BaseType::needsGenericSubstitutions() const {
  const TyTy::BaseType *x = destructure();
  switch (getKind()) {
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::ISize:
  case TypeKind::Float:
  case TypeKind::Inferred:
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
    return false;
  case TypeKind::Projection: {
    const ProjectionType &p = *static_cast<const ProjectionType *>(x);
    return static_cast<const SubstitutionRef &>(p).needsSubstitution();
  }
  case TypeKind::Function: {
    const FunctionType &fun = *static_cast<const FunctionType *>(x);
    return static_cast<const SubstitutionRef &>(fun).needsSubstitution();
  }
  case TypeKind::ADT: {
    const ADTType &adt = *static_cast<const ADTType *>(this);
    return static_cast<const SubstitutionRef &>(adt).needsSubstitution();
  }
  case TypeKind::Closure: {
    const ClosureType &clos = *static_cast<const ClosureType *>(x);
    return static_cast<const SubstitutionRef &>(clos).needsSubstitution();
  }
  }
}

TypeVariable::TypeVariable(basic::NodeId id) : id(id) {
  TyCtx *context = rust_compiler::session::session->getTypeContext();
  if (!context->lookupType(id))
    assert(false);
}

TyTy::BaseType *TypeVariable::getType() const {
  TyCtx *context = rust_compiler::session::session->getTypeContext();
  if (auto type = context->lookupType(id))
    return *type;
  assert(false);
}

BaseType::BaseType(basic::NodeId ref, basic::NodeId ty_ref, TypeKind kind,
                   TypeIdentity ident, std::set<basic::NodeId> refs)
    : TypeBoundsMappings({}), reference(ref), typeReference(ty_ref), kind(kind),
      identity(ident), combined(refs) {}

BaseType::BaseType(basic::NodeId ref, basic::NodeId ty_ref, TypeKind kind,
                   TypeIdentity ident,
                   std::vector<TypeBoundPredicate> specified_bounds,
                   std::set<basic::NodeId> refs)
    : TypeBoundsMappings(specified_bounds), reference(ref),
      typeReference(ty_ref), kind(kind), identity(ident), combined(refs) {}

basic::NodeId BaseType::getReference() const { return reference; }

basic::NodeId BaseType::getTypeReference() const { return typeReference; }

void BaseType::setReference(basic::NodeId ref) { reference = ref; }

void BaseType::appendReference(basic::NodeId ref) { combined.insert(ref); }

BoolType::BoolType(basic::NodeId reference, std::set<basic::NodeId> refs)
    : BaseType(reference, reference, TypeKind::Bool, TypeIdentity::empty(),
               refs) {}

BoolType::BoolType(basic::NodeId reference, basic::NodeId typeRef,
                   std::set<basic::NodeId> refs)
    : BaseType(reference, typeRef, TypeKind::Bool, TypeIdentity::empty(),
               refs) {}

std::string BoolType::toString() const { return "bool"; }

CharType::CharType(basic::NodeId reference, std::set<basic::NodeId> refs)
    : BaseType(reference, reference, TypeKind::Char, TypeIdentity::empty(),
               refs) {}

CharType::CharType(basic::NodeId reference, basic::NodeId type,
                   std::set<basic::NodeId> refs)
    : BaseType(reference, type, TypeKind::Char, TypeIdentity::empty(), refs) {}

std::string CharType::toString() const { return "char"; }

FloatType::FloatType(basic::NodeId id, FloatKind kind,
                     std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::Float, TypeIdentity::empty(), refs),
      kind(kind) {}

FloatType::FloatType(basic::NodeId id, basic::NodeId type, FloatKind kind,
                     std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::Float, TypeIdentity::empty(), refs),
      kind(kind) {}

std::string FloatType::toString() const {
  switch (kind) {
  case FloatKind::F32:
    return "f32";
  case FloatKind::F64:
    return "f64";
  }
}

IntKind IntType::getIntKind() const { return kind; }

IntType::IntType(basic::NodeId id, IntKind kind, std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty(), refs), kind(kind) {
}

IntType::IntType(basic::NodeId id, basic::NodeId type, IntKind kind,
                 std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::Int, TypeIdentity::empty(), refs),
      kind(kind) {}

std::string IntType::toString() const {
  switch (kind) {
  case IntKind::I8:
    return "i8";
  case IntKind::I16:
    return "i16";
  case IntKind::I32:
    return "i32";
  case IntKind::I64:
    return "i64";
  case IntKind::I128:
    return "i28";
  }
}

ISizeType::ISizeType(basic::NodeId id, std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::ISize, TypeIdentity::empty(), refs) {}

ISizeType::ISizeType(basic::NodeId id, basic::NodeId type,
                     std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::ISize, TypeIdentity::empty(), refs) {}

std::string ISizeType::toString() const { return "isize"; }

NeverType::NeverType(basic::NodeId id, std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::Never, TypeIdentity::empty(), refs) {}

NeverType::NeverType(basic::NodeId id, basic::NodeId type,
                     std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::Never, TypeIdentity::empty(), refs) {}

std::string NeverType::toString() const { return "!"; }

UintType::UintType(basic::NodeId id, UintKind kind,
                   std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty(), refs),
      kind(kind) {}

UintType::UintType(basic::NodeId id, basic::NodeId type, UintKind kind,
                   std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::Uint, TypeIdentity::empty(), refs),
      kind(kind) {}

std::string UintType::toString() const {
  switch (kind) {
  case UintKind::U8:
    return "u8";
  case UintKind::U16:
    return "u16";
  case UintKind::U32:
    return "u32";
  case UintKind::U64:
    return "u64";
  case UintKind::U128:
    return "u128";
  }
}

USizeType::USizeType(basic::NodeId id, std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::USize, TypeIdentity::empty(), refs) {}

USizeType::USizeType(basic::NodeId id, basic::NodeId type,
                     std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::USize, TypeIdentity::empty(), refs) {}

std::string USizeType::toString() const { return "usize"; }

StrType::StrType(basic::NodeId reference, std::set<basic::NodeId> refs)
    : BaseType(reference, reference, TypeKind::Str, TypeIdentity::empty(),
               refs) {}

StrType::StrType(basic::NodeId reference, basic::NodeId type,
                 std::set<basic::NodeId> refs)
    : BaseType(reference, type, TypeKind::Str, TypeIdentity::empty(), refs) {}

std::string StrType::toString() const { return "str"; }

unsigned StrType::getNumberOfSpecifiedBounds() const { return 0; }

TupleType::TupleType(basic::NodeId id, Location loc,
                     std::vector<TyTy::TypeVariable> parameterTyps,
                     std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::Tuple, TypeIdentity::from(loc), refs),
      fields(parameterTyps) {}
TupleType::TupleType(basic::NodeId id, basic::NodeId type, Location loc,
                     std::vector<TyTy::TypeVariable> parameterTyps,
                     std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::Tuple, TypeIdentity::from(loc), refs),
      fields(parameterTyps) {}

TupleType *TupleType::getUnitType(basic::NodeId id) {
  return new TupleType(id, Location::getBuiltinLocation());
}

std::string TupleType::toString() const {
  std::string str;
  llvm::raw_string_ostream stream(str);

  stream << "(";

  for (unsigned i = 0; i < fields.size(); ++i) {
    stream << fields[i].getType()->toString();
    if (i + 1 < fields.size())
      stream << ", ";
  }

  stream << ")";

  return stream.str();
}

ErrorType::ErrorType(basic::NodeId id, std::set<basic::NodeId> refs)
    : BaseType(id, id, TypeKind::Error, TypeIdentity::empty(), refs) {}

ErrorType::ErrorType(basic::NodeId id, basic::NodeId type,
                     std::set<basic::NodeId> refs)
    : BaseType(id, type, TypeKind::Error, TypeIdentity::empty(), refs) {}

std::string ErrorType::toString() const { return "error"; }

TyTy::BaseType *FunctionType::getReturnType() const { return returnType; }

std::string FunctionType::toString() const { assert(false); }

unsigned NeverType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned CharType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned ISizeType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned USizeType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned BoolType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned FloatType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned IntType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned UintType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned TupleType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned FunctionType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned ErrorType::getNumberOfSpecifiedBounds() const { return 0; }

unsigned InferType::getNumberOfSpecifiedBounds() const { return 0; }

InferType::InferType(basic::NodeId ref, InferKind kind, TypeHint hint,
                     Location loc, std::set<basic::NodeId> refs)
    : BaseType(ref, ref, TypeKind::Inferred, TypeIdentity::from(loc), refs),
      inferKind(kind), defaultHint(hint), loc(loc) {}
InferType::InferType(basic::NodeId ref, basic::NodeId type, InferKind kind,
                     TypeHint hint, Location loc, std::set<basic::NodeId> refs)
    : BaseType(ref, type, TypeKind::Inferred, TypeIdentity::from(loc), refs),
      inferKind(kind), defaultHint(hint), loc(loc) {}

std::string InferType::toString() const {
  switch (inferKind) {
  case InferKind::Float:
    return "<float>";
  case InferKind::Integral:
    return "<integer>";
  case InferKind::General:
    return "T?";
  }
}

bool isSignedIntegerLike(TypeKind kind) {
  switch (kind) {
  case TypeKind::Int:
  case TypeKind::ISize:
    return true;
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::Float:
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Closure:
  case TypeKind::Function:
  case TypeKind::Inferred:
  case TypeKind::Never:
  case TypeKind::Str:
  case TypeKind::Tuple:
  case TypeKind::Parameter:
  case TypeKind::ADT:
  case TypeKind::Error:
  case TypeKind::Array:
  case TypeKind::Projection:
  case TypeKind::Dynamic:
  case TypeKind::FunctionPointer:
  case TypeKind::PlaceHolder:
  case TypeKind::Slice:
  case TypeKind::RawPointer:
  case TypeKind::Reference:
    return false;
  }
}

bool isIntegerLike(TypeKind kind) {
  switch (kind) {
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::ISize:
    return true;
  case TypeKind::Float:
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Closure:
  case TypeKind::Function:
  case TypeKind::Inferred:
  case TypeKind::Never:
  case TypeKind::Str:
  case TypeKind::Tuple:
  case TypeKind::Parameter:
  case TypeKind::ADT:
  case TypeKind::Error:
  case TypeKind::Array:
  case TypeKind::Projection:
  case TypeKind::Dynamic:
  case TypeKind::FunctionPointer:
  case TypeKind::PlaceHolder:
  case TypeKind::Slice:
  case TypeKind::RawPointer:
  case TypeKind::Reference:
    return false;
  }
}

bool isFloatLike(TypeKind kind) {
  switch (kind) {
  case TypeKind::Float:
    return true;
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::ISize:
  case TypeKind::Closure:
  case TypeKind::Function:
  case TypeKind::Inferred:
  case TypeKind::Never:
  case TypeKind::Str:
  case TypeKind::Tuple:
  case TypeKind::Parameter:
  case TypeKind::ADT:
  case TypeKind::Error:
  case TypeKind::Array:
  case TypeKind::Projection:
  case TypeKind::FunctionPointer:
  case TypeKind::Dynamic:
  case TypeKind::PlaceHolder:
  case TypeKind::Slice:
  case TypeKind::RawPointer:
  case TypeKind::Reference:
    return false;
  }
}

std::string ClosureType::toString() const {
  std::stringstream s;
  s << "|" << parameters->toString() << "| {"
    << resultType.getType()->toString() << "}";

  return s.str();
}

unsigned ClosureType::getNumberOfSpecifiedBounds() const { return 0; }

StructFieldType::StructFieldType(basic::NodeId ref, const adt::Identifier &id,
                                 TyTy::BaseType *type, Location loc)
    : ref(ref), type(type), loc(loc), identifier(id) {}

VariantDef::VariantDef(basic::NodeId id, const adt::Identifier &identifier,
                       TypeIdentity ident, ast::Expression *discriminant)
    : id(id), identifier(identifier), ident(ident), discriminant(discriminant) {
}

VariantDef::VariantDef(basic::NodeId id, const adt::Identifier &identifier,
                       TypeIdentity ident, VariantKind kind,
                       ast::Expression *discriminant,
                       std::vector<TyTy::StructFieldType *> fields)
    : id(id), identifier(identifier), ident(ident), kind(kind),
      discriminant(discriminant), fields(fields) {}

std::string ADTType::toString() const {
  std::string variantsBuffer;
  for (size_t i = 0; i < variants.size(); ++i) {
    [[maybe_unused]] TyTy::VariantDef *variant = variants[i];
    // FIXME: variantsBuffer += variant->toString();
    if ((i + 1) < variants.size())
      variantsBuffer += ", ";
  }

  return identifier.toString() + substToString() + "{" + variantsBuffer + "}";
}

unsigned ADTType::getNumberOfSpecifiedBounds() const { return 0; }

std::string ArrayType::toString() const {
  return "[" + getElementType()->toString() + ":" + "CAPACITY" + "]";
}

unsigned ArrayType::getNumberOfSpecifiedBounds() const { return 0; }

TyTy::BaseType *ArrayType::getElementType() const { return type.getType(); }

unsigned ParamType::getNumberOfSpecifiedBounds() const { return bounds.size(); }

std::string ParamType::toString() const {
  assert(false && "to be implemented");
}

// SubstitutionArgumentMappings &FunctionType::getSubstitutionArguments() {
//   return usedArguments;
// }

bool BaseType::isConcrete() const {
  assert(false);

  switch (getKind()) {
  case TypeKind::Parameter:
  case TypeKind::Projection:
    return false;
  case TypeKind::PlaceHolder:
    return true;
  case TypeKind::Function: {
    const FunctionType *fun = static_cast<const FunctionType *>(this);
    for (const auto &param : fun->getParameters())
      if (!param.second->isConcrete())
        return false;
    return fun->getReturnType()->isConcrete();
  }
  case TypeKind::FunctionPointer: {
    const FunctionPointerType *fun =
        static_cast<const FunctionPointerType *>(this);
    for (const auto &param : fun->getParameters()) {
      const BaseType *p = param.getType();
      if (!p->isConcrete())
        return false;
    }
    return fun->getReturnType()->isConcrete();
  }
  case TypeKind::ADT: {
    const ADTType *adt = static_cast<const ADTType *>(this);
    if (adt->isUnit())
      return !adt->needsSubstitution();
    for (auto &variant : adt->getVariants()) {
      if (variant->getKind() == VariantKind::Enum)
        continue;
      for (auto &field : variant->getFields()) {
        const BaseType *fieldType = field->getFieldType();
        if (!fieldType->isConcrete())
          return false;
      }
    }
    return true;
  }
  case TypeKind::Array: {
    const ArrayType *array = static_cast<const ArrayType *>(this);
    return array->getElementType()->isConcrete();
  }
  case TypeKind::Slice: {
    const SliceType *slice = static_cast<const SliceType *>(this);
    return slice->getElementType()->isConcrete();
  }
  case TypeKind::RawPointer: {
    const RawPointerType *raw = static_cast<const RawPointerType *>(this);
    return raw->getBase()->isConcrete();
  }
  case TypeKind::Closure: {
    const ClosureType *clos = static_cast<const ClosureType *>(this);
    if (clos->getParameters()->isConcrete())
      return false;
    return clos->getResultType()->isConcrete();
  }
  case TypeKind::Tuple: {
    const TupleType *tuple = static_cast<const TupleType *>(this);
    for (size_t i = 0; i < tuple->getNumberOfFields(); ++i)
      if (!tuple->getField(i)->isConcrete())
        return false;
    return true;
  }
  case TypeKind::Reference: {
    const ReferenceType *ref = static_cast<const ReferenceType *>(this);
    return ref->getBase()->isConcrete();
  }
  case TypeKind::Inferred:
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::Float:
  case TypeKind::USize:
  case TypeKind::ISize:
  case TypeKind::Never:
  case TypeKind::Str:
  case TypeKind::Dynamic:
  case TypeKind::Error:
    return true;
  }
}

void InferType::applyScalarTypeHint(const BaseType &hint) {
  switch (hint.getKind()) {
  case TypeKind::USize:
  case TypeKind::ISize:
    inferKind = InferKind::Integral;
    defaultHint.kind = hint.getKind();
    break;
  case TypeKind::Int: {
    inferKind = InferKind::Integral;
    defaultHint.kind = hint.getKind();
    defaultHint.signHint = SignedHint::Signed;
    const IntType &i = static_cast<const IntType &>(hint);
    switch (i.getIntKind()) {
    case IntKind::I8:
      defaultHint.sizeHint = SizeHint::S8;
      break;
    case IntKind::I16:
      defaultHint.sizeHint = SizeHint::S16;
      break;
    case IntKind::I32:
      defaultHint.sizeHint = SizeHint::S32;
      break;
    case IntKind::I64:
      defaultHint.sizeHint = SizeHint::S64;
      break;
    case IntKind::I128:
      defaultHint.sizeHint = SizeHint::S128;
      break;
    }
    break;
  }
  case TypeKind::Uint: {
    inferKind = InferKind::Integral;
    defaultHint.kind = hint.getKind();
    defaultHint.signHint = SignedHint::Unsigned;
    const UintType &ui = static_cast<const UintType &>(hint);
    switch (ui.getUintKind()) {
    case UintKind::U8:
      defaultHint.sizeHint = SizeHint::S8;
      break;
    case UintKind::U16:
      defaultHint.sizeHint = SizeHint::S16;
      break;
    case UintKind::U32:
      defaultHint.sizeHint = SizeHint::S32;
      break;
    case UintKind::U64:
      defaultHint.sizeHint = SizeHint::S64;
      break;
    case UintKind::U128:
      defaultHint.sizeHint = SizeHint::S128;
      break;
    }
    break;
  }
  case TypeKind::Float: {
    inferKind = InferKind::Float;
    defaultHint.signHint = SignedHint::Signed;
    defaultHint.kind = hint.getKind();
    const FloatType &fl = static_cast<const FloatType &>(hint);
    switch (fl.getFloatKind()) {
    case FloatKind::F32:
      defaultHint.sizeHint = SizeHint::S32;
      break;
    case FloatKind::F64:
      defaultHint.sizeHint = SizeHint::S64;
      break;
    }
    break;
  }
  default:
    break;
  }
}

const BaseType *BaseType::destructure() const {
  uint32_t steps = 0;
  const BaseType *x = this;

  while (true) {
    if (steps++ >= MAX_RECURSION_DEPTH) {
      // report error
      return new ErrorType(getReference());
    }
    switch (x->getKind()) {
    case TypeKind::Parameter: {
      const ParamType *p = static_cast<const ParamType *>(x);
      BaseType *pr = p->resolve();
      if (pr == x)
        return pr;
      x = pr;
      break;
    }
    case TypeKind::PlaceHolder: {
      const PlaceholderType *p = static_cast<const PlaceholderType *>(x);
      if (!p->canResolve())
        return p;

      x = p->resolve();
      break;
    }
    case TypeKind::Projection: {
      const ProjectionType *p = static_cast<const ProjectionType *>(x);
      x = p->get();
      break;
    }
    default:
      return x;
    }
  }
}

BaseType *BaseType::destructure() {
  uint32_t steps = 0;
  BaseType *x = this;

  while (true) {
    if (steps++ >= MAX_RECURSION_DEPTH) {
      // report error
      return new ErrorType(getReference());
    }
    switch (x->getKind()) {
    case TypeKind::Parameter: {
      ParamType *p = static_cast<ParamType *>(x);
      BaseType *pr = p->resolve();
      if (pr == x)
        return pr;
      x = pr;
      break;
    }
    case TypeKind::PlaceHolder: {
      PlaceholderType *p = static_cast<PlaceholderType *>(x);
      if (!p->canResolve())
        return p;

      x = p->resolve();
      break;
    }
    case TypeKind::Projection: {
      ProjectionType *p = static_cast<ProjectionType *>(x);
      x = p->get();
      break;
    }
    default:
      return x;
    }
  }
}

bool PlaceholderType::canResolve() const {
  TyCtx *context = rust_compiler::session::session->getTypeContext();
  return context->lookupAssociatedTypeMapping(getTypeReference()).has_value();
}

BaseType *PlaceholderType::resolve() const {
  TyCtx *context = rust_compiler::session::session->getTypeContext();
  std::optional<NodeId> result =
      context->lookupAssociatedTypeMapping(getTypeReference());
  assert(result.has_value());

  return TypeVariable(*result).getType();
}

BaseType *ParamType::resolve() const {
  TypeVariable var = {getTypeReference()};
  BaseType *r = var.getType();

  while (r->getKind() == TypeKind::Parameter) {
    ParamType *rr = static_cast<ParamType *>(r);
    if (!rr->canResolve())
      break;

    TypeVariable v(rr->getTypeReference());
    BaseType *n = v.getType();

    // fix infinite loop
    if (r == n)
      break;

    r = n;
  }

  if (r->getKind() == TypeKind::Parameter &&
      (r->getReference() == r->getTypeReference()))
    return TypeVariable(r->getTypeReference()).getType();

  return r;
}

void TypeBoundsMappings::addBound(const TypeBoundPredicate &predicate) {
  for (TypeBoundPredicate &bound : specifiedBounds) {
    if (bound.getId() == predicate.getId())
      return;
  }
  specifiedBounds.push_back(predicate);
}

BaseType *BoolType::clone() const {
  return new BoolType(getReference(), getTypeReference(),
                      getCombinedReferences());
}

BaseType *IntType::clone() const {
  return new IntType(getReference(), getTypeReference(), getIntKind(),
                     getCombinedReferences());
}

BaseType *UintType::clone() const {
  return new UintType(getReference(), getTypeReference(), getUintKind(),
                      getCombinedReferences());
}

BaseType *FloatType::clone() const {
  return new FloatType(getReference(), getTypeReference(), getFloatKind(),
                       getCombinedReferences());
}

BaseType *USizeType::clone() const {
  return new USizeType(getReference(), getTypeReference(),
                       getCombinedReferences());
}

BaseType *ISizeType::clone() const {
  return new ISizeType(getReference(), getTypeReference(),
                       getCombinedReferences());
}

BaseType *CharType::clone() const {
  return new CharType(getReference(), getTypeReference(),
                      getCombinedReferences());
}

BaseType *StrType::clone() const {
  return new StrType(getReference(), getTypeReference(),
                     getCombinedReferences());
}

BaseType *NeverType::clone() const {
  return new NeverType(getReference(), getTypeReference(),
                       getCombinedReferences());
}

BaseType *TupleType::clone() const {
  std::vector<TypeVariable> clonedFields;
  for (const auto &f : fields)
    clonedFields.push_back(f.clone());

  return new TupleType(getReference(), getTypeReference(),
                       Location::getEmptyLocation(), clonedFields,
                       getCombinedReferences());
}

BaseType *FunctionType::clone() const {

  std::vector<
      std::pair<std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
                BaseType *>>
      clonedParams;
  for (auto &p : parameters)
    clonedParams.push_back({p.first, p.second->clone()});

  return new FunctionType(getReference(), getTypeReference(), getIdentifier(),
                          getTypeIdentity(), flags, clonedParams,
                          getReturnType()->clone(), cloneSubsts(),
                          getCombinedReferences());
}

BaseType *ClosureType::clone() const {
  return new ClosureType(getReference(), getTypeReference(), getTypeIdentity(),
                         (TyTy::TupleType *)parameters->clone(), resultType,
                         cloneSubsts(), captures, getCombinedReferences(),
                         getSpecifiedBounds());
}

BaseType *ADTType::clone() const {
  std::vector<VariantDef *> clonedVariants;
  for (auto &variant : variants)
    clonedVariants.push_back(variant->clone());

  return new ADTType(getReference(), getTypeReference(), identifier,
                     getTypeIdentity(), kind, clonedVariants, cloneSubsts(),
                     usedArguments, getCombinedReferences());
}

BaseType *ArrayType::clone() const {
  return new ArrayType(getReference(), getTypeReference(), loc, expr, type,
                       getCombinedReferences());
}

BaseType *ParamType::clone() const {
  return new ParamType(identifier, loc, getReference(), getTypeReference(),
                       type, bounds, getCombinedReferences());
}

BaseType *ErrorType::clone() const {
  return new ErrorType(getReference(), getTypeReference(),
                       getCombinedReferences());
}

TypeVariable TypeVariable::clone() const {
  BaseType *c = getType()->clone();
  return TypeVariable(c->getReference());
}

VariantDef *VariantDef::clone() const {
  std::vector<StructFieldType *> clonedFields;
  for (auto &f : fields)
    clonedFields.push_back((StructFieldType *)f->clone());

  return new VariantDef(id, identifier, ident, kind, discriminant, fields);
}

StructFieldType *StructFieldType::clone() const {
  return new StructFieldType(ref, identifier, type->clone(), loc);
}

BaseType *InferType::clone() const {
  TyCtx *context = rust_compiler::session::session->getTypeContext();

  InferType *cloned =
      new InferType(rust_compiler::basic::getNextNodeId(), inferKind,
                    defaultHint, loc, getCombinedReferences());

  context->insertType(
      NodeIdentity(cloned->getReference(),
                   rust_compiler::session::session->getCurrentCrateNum(), loc),
      cloned);

  context->insertLocation(cloned->getReference(),
                          context->lookupLocation(getReference()));

  cloned->appendReference(getReference());

  return cloned;
}

void ClosureType::setupFnOnceOutput() const {
  // TyCtx *context = rust_compiler::session::session->getTypeContext();

  assert(false);

  //  std::optional<ast::Item *> item = context->lookupItem(traitId);
  //  assert(item.has_value());
  //  assert((*item)->getItemKind() == ItemKind::VisItem);
  //  ast::VisItem *visItem = static_cast<ast::VisItem*>(*item);
  //  assert(visItem->getKind() == VisItemKind::Trait);
  //  ast::Trait *trait = static_cast<ast::Trait*>(visItem);
}

TraitReference *TypeBoundPredicate::get() const {
  TyCtx *context = rust_compiler::session::session->getTypeContext();
  std::optional<TraitReference *> ref = context->lookupTraitReference(id);
  assert(ref.has_value());
  return *ref;
}

FunctionType *
FunctionType::handleSubstitions(SubstitutionArgumentMappings &mappings) {
  FunctionType *fn = static_cast<FunctionType *>(clone());
  fn->setTypeReference(basic::getNextNodeId());
  fn->usedArguments = mappings;

  for (auto &sub : fn->getSubstitutions()) {
    std::optional<SubstitutionArg> arg =
        mappings.getArgumentForSymbol(sub.getParamType());
    if (arg)
      sub.fillParamType(mappings, mappings.getLocation());
  }

  BaseType *ret = fn->getReturnType();
  bool isParamType = ret->getKind() == TypeKind::Parameter;
  if (isParamType) {
    ParamType *p = static_cast<ParamType *>(ret);
    std::optional<SubstitutionArg> arg = mappings.getArgumentForSymbol(p);
    if (arg) {
      if ((*arg).getType()->getKind() == TypeKind::Parameter ||
          (*arg).getType()->getKind() != TypeKind::Inferred) {
        BaseType *newField = (*arg).getType()->clone();
        newField->setReference(ret->getReference());
        fn->returnType = newField;
      } else {
        ret->setTypeReference((*arg).getType()->getReference());
      }
    }
  } else if (ret->needsGenericSubstitutions() || !ret->isConcrete()) {
    InternalSubstitutionsMapper mapper;
    BaseType *concrete = mapper.resolve(ret, mappings);
    if (concrete == nullptr || concrete->getKind() == TypeKind::Error) {
      // report error
      assert(false);
      return nullptr;
    }
    BaseType *newField = concrete->clone();
    newField->setReference(ret->getReference());
    fn->returnType = newField;
  }

  for (auto &param : fn->getParameters()) {
    BaseType *paramType = param.second;
    if (paramType->getKind() == TypeKind::Parameter) {
      ParamType *p = static_cast<ParamType *>(paramType);

      std::optional<SubstitutionArg> arg = mappings.getArgumentForSymbol(p);
      if (arg) {
        BaseType *argType = (*arg).getType();
        if (argType->getKind() == TypeKind::Parameter ||
            argType->getKind() != TypeKind::Inferred) {
          BaseType *newField = argType->clone();
          newField->setReference(paramType->getReference());
          param.second = newField;
        }
      }
    } else if (paramType->hasSubsititionsDefined() ||
               !paramType->isConcrete()) {
      InternalSubstitutionsMapper mapper;
      BaseType *concrete = mapper.resolve(paramType, mappings);
      if (concrete == nullptr || concrete->getKind() == TypeKind::Error) {
        // report error
        assert(false);
      }

      BaseType *newField = concrete->clone();
      newField->setReference(paramType->getReference());
      param.second = newField;
    }
  }

  return fn;
}

bool ParamType::isImplicitSelfTrait() const { return isTraitSelf; }

bool BaseType::hasSubsititionsDefined() const {
  const BaseType *x = destructure();
  switch (x->getKind()) {
  case TypeKind::Inferred:
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::ISize:
  case TypeKind::Never:
  case TypeKind::Str:
  case TypeKind::Float:
  case TypeKind::Dynamic:
  case TypeKind::Error:
  case TypeKind::FunctionPointer:
  case TypeKind::Array:
  case TypeKind::Slice:
  case TypeKind::RawPointer:
  case TypeKind::Reference:
  case TypeKind::Tuple:
  case TypeKind::Parameter:
  case TypeKind::PlaceHolder:
    return false;
  case TypeKind::Projection: {
    const ProjectionType &p = *static_cast<const ProjectionType *>(x);
    const SubstitutionRef &ref = static_cast<const SubstitutionRef &>(p);
    return ref.hasSubstitutions();
  }
  case TypeKind::Function: {
    const FunctionType &p = *static_cast<const FunctionType *>(x);
    const SubstitutionRef &ref = static_cast<const SubstitutionRef &>(p);
    return ref.hasSubstitutions();
  }
  case TypeKind::ADT: {
    const ADTType &p = *static_cast<const ADTType *>(x);
    const SubstitutionRef &ref = static_cast<const SubstitutionRef &>(p);
    return ref.hasSubstitutions();
  }
  case TypeKind::Closure: {
    const ClosureType &p = *static_cast<const ClosureType *>(x);
    const SubstitutionRef &ref = static_cast<const SubstitutionRef &>(p);
    return ref.hasSubstitutions();
  }
  }
  return false;
}

bool BaseType::isBoundsCompatible(const BaseType &other, Location loc,
                                  bool emitError) const {
  std::vector<std::reference_wrapper<const TypeBoundPredicate>>
      unsatisfiedBounds;
  for (auto bound : getSpecifiedBounds()) {
    if (!other.satisfiesBound(bound))
      unsatisfiedBounds.push_back(bound);
  }

  if (unsatisfiedBounds.size() > 0) {
    // report error
    assert(false);
  }

  return unsatisfiedBounds.size() == 0;
}

ClosureType *
ClosureType::handleSubstitions(SubstitutionArgumentMappings &mappings) {
  llvm_unreachable("not supported");
  return nullptr;
}

TypeBoundPredicate::TypeBoundPredicate(TraitReference &ref, Location loc)
    : SubstitutionRef({}, SubstitutionArgumentMappings::empty()),
      id(ref.getNodeId()), loc(loc) {
  substitutions.clear();
  for (const auto &p : ref.getTraitSubsts())
    substitutions.push_back(p.clone());

  SubstitutionArg placeholder(&getSubstitutions().front(), nullptr);
  usedArguments.getMappings().push_back(placeholder);
}

TypeBoundPredicate::TypeBoundPredicate(
    basic::NodeId id, std::vector<SubstitutionParamMapping> subst, Location loc)
    : SubstitutionRef({}, SubstitutionArgumentMappings::empty()), id(id),
      loc(loc) {
  substitutions.clear();
  for (const auto &p : subst)
    substitutions.push_back(p.clone());

  SubstitutionArg placeholder(&getSubstitutions().front(), nullptr);
  usedArguments.getMappings().push_back(placeholder);
}

void BaseType::inheritBounds(const BaseType &other) {
  inheritBounds(other.getSpecifiedBounds());
}

void BaseType::inheritBounds(
    std::vector<TyTy::TypeBoundPredicate> specifiedBounds) {
  for (TyTy::TypeBoundPredicate &pred : specifiedBounds) {
    addBound(pred);
  }
}

std::string DynamicObjectType::toString() const {
  return "dyn [" + rawBoundsToString() + "]";
}

std::string TypeBoundsMappings::rawBoundsToString() const {
  std::string buf;
  for (size_t i = 0; i < specifiedBounds.size(); i++) {
    const TypeBoundPredicate &b = specifiedBounds.at(i);
    bool has_next = (i + 1) < specifiedBounds.size();
    buf += b.toString() + (has_next ? " + " : "");
  }
  return buf;
}

std::string TypeBoundPredicate::toString() const {
  return get()->toString() + substToString();
}

unsigned DynamicObjectType::getNumberOfSpecifiedBounds() const { return 0; }

BaseType *DynamicObjectType::clone() const {
  return new DynamicObjectType(getReference(), getTypeReference(),
                               getTypeIdentity(), getSpecifiedBounds(),
                               getCombinedReferences());
}

ADTType *ADTType::handleSubstitions(SubstitutionArgumentMappings &mappings) {
  assert(false);
}

bool BaseType::satisfiesBound(const TypeBoundPredicate &predicate) const {
  assert(false);
}

std::string RawPointerType::toString() const {
  return std::string("* ") + (isMutable() ? "mut" : "const") + " " +
         getBase()->toString();
}

unsigned RawPointerType::getNumberOfSpecifiedBounds() const { return 0; }

BaseType *RawPointerType::clone() const {
  return new RawPointerType(getReference(), getTypeReference(), base, mut,
                            getCombinedReferences());
}

bool TypeBoundPredicate::requiresGenericArgs() const {
  if (isError())
    return false;
  return substitutions.size() > 1;
}

void TypeBoundPredicate::applyGenericArgs(ast::GenericArgs *genericArgs,
                                          bool hasAssociatedSelf) {
  assert(!substitutions.empty());
  if (hasAssociatedSelf)
    usedArguments = SubstitutionArgumentMappings::empty();
  else
    assert(!usedArguments.isEmpty());

  usedArguments = getMappingsFromGenericArgs(*genericArgs);

  errorFlag |= usedArguments.isError();

  SubstitutionArgumentMappings &substMappings = usedArguments;

  for (SubstitutionParamMapping &param : getSubstitutions()) {
    std::optional<SubstitutionArg> arg =
        substMappings.getArgumentForSymbol(param.getParamType());
    if (arg && (*arg).getType() != nullptr)
      param.fillParamType(substMappings, substMappings.getLocation());
  }

  for (auto &it : substMappings.getBindingArgs()) {
    TypeBoundPredicateItem item = lookupAssociatedItem(it.first);
    assert(!item.isError());

    const TraitItemReference *itemRef = item.getRawItem();
    itemRef->associatedTypeSet(it.second);
  }
}

size_t TypeBoundPredicate::getNumberOfAssociatedBindings() const {
  size_t count = 0;
  TraitReference *traitRef = get();
  for (const TraitItemReference &traitItem : traitRef->getTraitItems())
    if (traitItem.isType())
      ++count;
  return count;
}

TypeBoundPredicateItem
TypeBoundPredicate::lookupAssociatedItem(const Identifier &id) const {
  TraitReference *ref = get();
  std::optional<TraitItemReference *> traitRef = ref->lookupTraitItem(id);
  if (!traitRef)
    return TypeBoundPredicateItem::error();

  return TypeBoundPredicateItem(this, *traitRef);
}

void PlaceholderType::setAssociatedType(basic::NodeId id) {
  rust_compiler::session::session->getTypeContext()
      ->insertAssociatedTypeMapping(getTypeReference(), id);
}

TypeVariable TypeVariable::getImplicitInferVariable(Location loc) {
  InferType *infer =
      new InferType(rust_compiler::basic::getNextNodeId(), InferKind::General,
                    TypeHint::unknown(), loc);

  TyCtx *ctx = rust_compiler::session::session->getTypeContext();
  ctx->insertType(
      NodeIdentity(infer->getReference(),
                   rust_compiler::session::session->getCurrentCrateNum(), loc),
      infer);
  ctx->insertLocation(infer->getReference(), loc);

  return TypeVariable(infer->getReference());
}

std::optional<TyTy::StrType *> ReferenceType::isDynStrType() const {
  TyTy::BaseType *element = getBase()->destructure();
  if (element->getKind() != TypeKind::Str)
    return std::nullopt;
  //  if (str == nullptr)
  //    return true;

  return static_cast<TyTy::StrType *>(element);
}

std::string ReferenceType::toString() const {
  return std::string("&") + (isMutable() ? "mut" : "") + " " +
         getBase()->toString();
}

BaseType *ReferenceType::clone() const {
  return new ReferenceType(getReference(), getTypeReference(), base, mut,
                           getCombinedReferences());
}

bool ErrorType::canEqual(const TyTy::BaseType *other, bool emitErrors) const {
  return getKind() == other->getKind();
}

bool ADTType::canEqual(const BaseType *other, bool emitErrors) const {
  ADTCmp r(this);
  return r.canEqual(other);
}

bool TupleType::canEqual(const BaseType *other, bool emitErrors) const {
  TupleCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool FunctionType::canEqual(const BaseType *other, bool emitErrors) const {
  FunctionCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool FunctionPointerType::canEqual(const BaseType *other,
                                   bool emitErrors) const {
  FunctionPointerCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool ClosureType::canEqual(const BaseType *other, bool emitErrors) const {
  ClosureCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool ArrayType::canEqual(const BaseType *other, bool emitErrors) const {
  ArrayCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool SliceType::canEqual(const BaseType *other, bool emitErrors) const {
  SliceCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool BoolType::canEqual(const BaseType *other, bool emitErrors) const {
  BoolCmp r;
  return r.canEqual(other);
}

bool IntType::canEqual(const BaseType *other, bool emitErrors) const {
  IntCmp r(this);
  return r.canEqual(other);
}

bool UintType::canEqual(const BaseType *other, bool emitErrors) const {
  UintCmp r(this);
  return r.canEqual(other);
}

bool FloatType::canEqual(const BaseType *other, bool emitErrors) const {
  FloatCmp r(this);
  return r.canEqual(other);
}

bool USizeType::canEqual(const BaseType *other, bool emitErrors) const {
  USizeCmp r;
  return r.canEqual(other);
}

bool ISizeType::canEqual(const BaseType *other, bool emitErrors) const {
  ISizeCmp r;
  return r.canEqual(other);
}

bool CharType::canEqual(const BaseType *other, bool emitErrors) const {
  CharCmp r;
  return r.canEqual(other);
}

bool ReferenceType::canEqual(const BaseType *other, bool emitErrors) const {
  ReferenceCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool RawPointerType::canEqual(const BaseType *other, bool emitErrors) const {
  RawPointerCmp r(this, emitErrors);
  return r.canEqual(other);
}

bool ParamType::canEqual(const BaseType *other, bool emitErrors) const {
  ParamCmp r(this);
  return r.canEqual(other);
}

bool StrType::canEqual(const BaseType *other, bool emitErrors) const {
  StrCmp r;
  return r.canEqual(other);
}

bool NeverType::canEqual(const BaseType *other, bool emitErrors) const {
  NeverCmp r;
  return r.canEqual(other);
}

bool PlaceholderType::canEqual(const BaseType *other, bool emitErrors) const {
  PlaceholderCmp r;
  return r.canEqual(other);
}

bool ProjectionType::canEqual(const BaseType *other, bool emitErrors) const {
  return base->canEqual(other, emitErrors);
}

bool DynamicObjectType::canEqual(const BaseType *other, bool emitErrors) const {
  DynamicCmp r(this);
  return r.canEqual(other);
}

bool InferType::canEqual(const BaseType *other, bool emitErrors) const {
  InferCmp r(this);
  return r.canEqual(other);
}

bool BaseType::boundsCompatible(const TyTy::BaseType *other, Location loc,
                                bool emitError) const {
  std::vector<std::reference_wrapper<const TypeBoundPredicate>>
      unsatisfiedBounds;

  for (auto &bound : getSpecifiedBounds())
    if (!other->satisfiesBound(bound))
      unsatisfiedBounds.push_back(bound);

  if (unsatisfiedBounds.size() > 0) {
    // report error
    assert(false);
  }

  return unsatisfiedBounds.size() == 1;
}

std::string SliceType::toString() const {
  return "[" + getElementType()->toString() + "]";
}

TyTy::BaseType *SliceType::clone() const {
  return new SliceType(getReference(), getTypeReference(), getLocation(),
                       elementType.clone(), getCombinedReferences());
}

bool VariantDef::lookupField(lexer::Identifier ident,
                             TyTy::StructFieldType **lookup, size_t *index) {
  size_t i = 0;
  for (auto &field : fields) {
    if (field->getName() == ident) {
      if (index != nullptr)
        *index = i;

      if (lookup != nullptr)
        *lookup = field;

      return true;
    }
    ++i;
  }

  return false;
}

std::string TypeKind2String(TypeKind kind) {
  switch (kind) {
  case TypeKind::Bool:
    return "bool";
  case TypeKind::Char:
    return "char";
  case TypeKind::Int:
    return "int";
  case TypeKind::Uint:
    return "uint";
  case TypeKind::USize:
    return "usize";
  case TypeKind::ISize:
    return "isize";
  case TypeKind::Float:
    return "float";
  case TypeKind::Closure:
    return "closure";
  case TypeKind::Function:
    return "function";
  case TypeKind::Inferred:
    return "inferred";
  case TypeKind::Never:
    return "never";
  case TypeKind::Str:
    return "str";
  case TypeKind::Tuple:
    return "tuple";
  case TypeKind::Parameter:
    return "parameter";
  case TypeKind::ADT:
    return "adt";
  case TypeKind::Array:
    return "array";
  case TypeKind::Slice:
    return "slice";
  case TypeKind::Projection:
    return "projection";
  case TypeKind::Dynamic:
    return "dynamic";
  case TypeKind::PlaceHolder:
    return "placeholder";
  case TypeKind::FunctionPointer:
    return "functionpointer";
  case TypeKind::RawPointer:
    return "rawpointer";
  case TypeKind::Reference:
    return "reference";
  case TypeKind::Error:
    return "error";
  }
}

} // namespace rust_compiler::tyctx::TyTy
