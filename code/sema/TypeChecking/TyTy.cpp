#include "TyTy.h"

#include "ADT/CanonicalPath.h"
#include "Basic/Ids.h"
#include "Location.h"
#include "TyCtx/TyCtx.h"
#include "TypeIdentity.h"

#include <llvm/Support/raw_ostream.h>

using namespace rust_compiler::adt;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking::TyTy {

TypeVariable::TypeVariable(basic::NodeId id) : id(id) {
  TyCtx *context = TyCtx::get();
  if (!context->lookupType(id))
    assert(false);
}

TyTy::BaseType *TypeVariable::getType() const {
  TyCtx *context = TyCtx::get();
  if (auto type = context->lookupType(id))
    return *type;
  assert(false);
}

BaseType::BaseType(basic::NodeId ref, basic::NodeId typeReference,
                   TypeKind kind, TypeIdentity ident)
    : reference(ref), typeReference(typeReference), kind(kind),
      identity(ident) {}

BaseType::~BaseType() {}

basic::NodeId BaseType::getReference() const { return reference; }

basic::NodeId BaseType::getTypeReference() const { return typeReference; }

void BaseType::setReference(basic::NodeId ref) { reference = ref; }

BoolType::BoolType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Bool, TypeIdentity::empty()) {}

bool BoolType::needsGenericSubstitutions() const { return false; }

std::string BoolType::toString() const { return "bool"; }

CharType::CharType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Char, TypeIdentity::empty()) {}

bool CharType::needsGenericSubstitutions() const { return false; }

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

bool FloatType::needsGenericSubstitutions() const { return false; }

IntType::IntType(basic::NodeId id, IntKind kind)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()), kind(kind) {}

bool IntType::needsGenericSubstitutions() const { return false; }

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
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()) {}

bool ISizeType::needsGenericSubstitutions() const { return false; }

std::string ISizeType::toString() const { return "isize"; }

NeverType::NeverType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Never, TypeIdentity::empty()) {}

bool NeverType::needsGenericSubstitutions() const { return false; }

std::string NeverType::toString() const { return "!"; }

UintType::UintType(basic::NodeId id, UintKind kind)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()), kind(kind) {}

bool UintType::needsGenericSubstitutions() const { return false; }

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
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()) {}

bool USizeType::needsGenericSubstitutions() const { return false; }

std::string USizeType::toString() const { return "usize"; }

StrType::StrType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Str, TypeIdentity::empty()) {}

bool StrType::needsGenericSubstitutions() const { return false; }

std::string StrType::toString() const { return "str"; }

unsigned StrType::getNumberOfSpecifiedBounds() { return 0; }

TupleType::TupleType(basic::NodeId id, Location loc)
    : BaseType(id, id, TypeKind::Tuple, TypeIdentity::from(loc)) {}

bool TupleType::needsGenericSubstitutions() const { return false; }

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

bool ErrorType::needsGenericSubstitutions() const { return false; }

std::string ErrorType::toString() const { return "error"; }

FunctionType::FunctionType(
    basic::NodeId id, std::string_view name, tyctx::ItemIdentity ident,
    std::vector<std::pair<
        std::shared_ptr<rust_compiler::ast::patterns::PatternNoTopAlt>,
        TyTy::BaseType *>>
        parameters,
    TyTy::BaseType *returnType,
    std::vector<TyTy::SubstitutionParamMapping> substitutions)
    : BaseType(id, id, TypeKind::Function,
               TypeIdentity(ident.getPath(), ident.getLocation())),
      id(id), name(name), ident(ident), parameters(parameters),
      returnType(returnType), substitutions(substitutions) {}

TyTy::BaseType *FunctionType::getReturnType() const { return returnType; }
bool FunctionType::needsGenericSubstitutions() const { return true; }

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

} // namespace rust_compiler::sema::type_checking::TyTy
