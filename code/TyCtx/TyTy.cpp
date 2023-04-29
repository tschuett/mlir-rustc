#include "TyCtx/TyTy.h"

#include "ADT/CanonicalPath.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "Location.h"
#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TypeIdentity.h"

#include <llvm/Support/raw_ostream.h>
#include <sstream>

using namespace rust_compiler::adt;
using namespace rust_compiler::tyctx;

namespace rust_compiler::tyctx::TyTy {

bool BaseType::needsGenericSubstitutions() const {
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
    assert(false);
  }
  case TypeKind::Function: {
    const FunctionType *fun = static_cast<const FunctionType *>(this);
    return static_cast<const SubstitutionsReference *>(fun)
        ->needsSubstitution();
  }
  case TypeKind::ADT: {
    const ADTType *adt = static_cast<const ADTType *>(this);
    return static_cast<const SubstitutionsReference *>(adt)
        ->needsSubstitution();
  }
  case TypeKind::Closure: {
    const ClosureType *clos = static_cast<const ClosureType *>(this);
    return static_cast<const ClosureType *>(clos)->needsSubstitution();
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

BaseType::BaseType(basic::NodeId ref, basic::NodeId typeReference,
                   TypeKind kind, TypeIdentity ident)
    : reference(ref), typeReference(typeReference), kind(kind),
      identity(ident) {}

basic::NodeId BaseType::getReference() const { return reference; }

basic::NodeId BaseType::getTypeReference() const { return typeReference; }

void BaseType::setReference(basic::NodeId ref) { reference = ref; }

void BaseType::appendReference(basic::NodeId ref) { combined.insert(ref); }

BoolType::BoolType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Bool, TypeIdentity::empty()) {}

std::string BoolType::toString() const { return "bool"; }

CharType::CharType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Char, TypeIdentity::empty()) {}

std::string CharType::toString() const { return "char"; }

FloatType::FloatType(basic::NodeId id, FloatKind kind)
    : BaseType(id, id, TypeKind::Float, TypeIdentity::empty()), kind(kind) {}

std::string FloatType::toString() const {
  switch (kind) {
  case FloatKind::F32:
    return "f32";
  case FloatKind::F64:
    return "f64";
  }
}

IntKind IntType::getIntKind() const { return kind; }

IntType::IntType(basic::NodeId id, IntKind kind)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()), kind(kind) {}

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

ISizeType::ISizeType(basic::NodeId id)
    : BaseType(id, id, TypeKind::ISize, TypeIdentity::empty()) {}

std::string ISizeType::toString() const { return "isize"; }

NeverType::NeverType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Never, TypeIdentity::empty()) {}

std::string NeverType::toString() const { return "!"; }

UintType::UintType(basic::NodeId id, UintKind kind)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()), kind(kind) {}

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

USizeType::USizeType(basic::NodeId id)
    : BaseType(id, id, TypeKind::USize, TypeIdentity::empty()) {}

std::string USizeType::toString() const { return "usize"; }

StrType::StrType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Str, TypeIdentity::empty()) {}

std::string StrType::toString() const { return "str"; }

unsigned StrType::getNumberOfSpecifiedBounds() { return 0; }

TupleType::TupleType(basic::NodeId id, Location loc,
                     std::span<TyTy::TypeVariable> parameterTyps)
    : BaseType(id, id, TypeKind::Tuple, TypeIdentity::from(loc)) {
  fields = {parameterTyps.begin(), parameterTyps.end()};
}

TupleType::TupleType(basic::NodeId id, Location loc)
    : BaseType(id, id, TypeKind::Tuple, TypeIdentity::from(loc)) {}

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

ErrorType::ErrorType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Error, TypeIdentity::empty()) {}

std::string ErrorType::toString() const { return "error"; }

FunctionType::FunctionType(
    basic::NodeId id, lexer::Identifier name, tyctx::ItemIdentity ident,
    std::vector<std::pair<
        std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
        TyTy::BaseType *>>
        parameters,
    TyTy::BaseType *returnType,
    std::vector<TyTy::SubstitutionParamMapping> substitutions)
    : BaseType(id, id, TypeKind::Function,
               TypeIdentity(ident.getPath(), ident.getLocation())),
      SubstitutionsReference(substitutions), id(id), name(name), ident(ident),
      parameters(parameters), returnType(returnType) {}

TyTy::BaseType *FunctionType::getReturnType() const { return returnType; }

std::string FunctionType::toString() const { assert(false); }

unsigned NeverType::getNumberOfSpecifiedBounds() { return 0; }

unsigned CharType::getNumberOfSpecifiedBounds() { return 0; }

unsigned ISizeType::getNumberOfSpecifiedBounds() { return 0; }

unsigned USizeType::getNumberOfSpecifiedBounds() { return 0; }

unsigned BoolType::getNumberOfSpecifiedBounds() { return 0; }

unsigned FloatType::getNumberOfSpecifiedBounds() { return 0; }

unsigned IntType::getNumberOfSpecifiedBounds() { return 0; }

unsigned UintType::getNumberOfSpecifiedBounds() { return 0; }

unsigned TupleType::getNumberOfSpecifiedBounds() { return 0; }

unsigned FunctionType::getNumberOfSpecifiedBounds() { return 0; }

unsigned ErrorType::getNumberOfSpecifiedBounds() { return 0; }

unsigned InferType::getNumberOfSpecifiedBounds() { return 0; }

InferType::InferType(basic::NodeId ref, InferKind kind, Location loc)
    : BaseType(ref, ref, TypeKind::Inferred, TypeIdentity::from(loc)),
      inferKind(kind) {}

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

unsigned ClosureType::getNumberOfSpecifiedBounds() { return 0; }

StructFieldType::StructFieldType(basic::NodeId ref, const adt::Identifier &id,
                                 TyTy::BaseType *type, Location loc)
    : ref(ref), type(type), loc(loc), id(id) {}

StructFieldType::StructFieldType(basic::NodeId ref, std::string_view id,
                                 TyTy::BaseType *type, Location loc)
    : ref(ref), type(type), loc(loc), id(id) {}

VariantDef::VariantDef(basic::NodeId id, const adt::Identifier &identifier,
                       TypeIdentity ident, VariantKind kind,
                       std::span<TyTy::StructFieldType *> f)
    : id(id), identifier(identifier), ident(ident), kind(kind) {
  fields = {f.begin(), f.end()};
}

ADTType::ADTType(basic::NodeId id, const adt::Identifier &identifier,
                 TypeIdentity ident, ADTKind kind,
                 std::span<VariantDef *> variant,
                 std::vector<SubstitutionParamMapping> sub)
    : BaseType(id, id, TypeKind::ADT, ident), SubstitutionsReference(sub),
      identifier(identifier), kind(kind) {
  variants = {variant.begin(), variant.end()};
}

std::string ADTType::toString() const {
  std::string variantsBuffer;
  for (size_t i = 0; i < variants.size(); ++i) {
    [[maybe_unused]] TyTy::VariantDef *variant = variants[i];
    // FIXME: variantsBuffer += variant->toString();
    if ((i + 1) < variants.size())
      variantsBuffer += ", ";
  }

  return /*identifier*/ substToString() + "{" + variantsBuffer + "}";
}

unsigned ADTType::getNumberOfSpecifiedBounds() { return 0; }

std::string ArrayType::toString() const {
  return "[" + getElementType()->toString() + ":" + "CAPACITY" + "]";
}

unsigned ArrayType::getNumberOfSpecifiedBounds() { return 0; }

TyTy::BaseType *ArrayType::getElementType() const { return type.getType(); }

unsigned ParamType::getNumberOfSpecifiedBounds() { return bounds.size(); }

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

} // namespace rust_compiler::tyctx::TyTy
