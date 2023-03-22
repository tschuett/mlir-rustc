#include "TyTy.h"

#include "ADT/CanonicalPath.h"
#include "Basic/Ids.h"
#include "Location.h"
#include "TypeIdentity.h"

using namespace rust_compiler::adt;

namespace rust_compiler::sema::type_checking::TyTy {

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

CharType::CharType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Char, TypeIdentity::empty()) {}

bool CharType::needsGenericSubstitutions() const { return false; }

FloatType::FloatType(basic::NodeId id, FloatKind kind)
    : BaseType(id, id, TypeKind::Float, TypeIdentity::empty()), kind(kind) {}

bool FloatType::needsGenericSubstitutions() const { return false; }

IntType::IntType(basic::NodeId id, IntKind kind)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()), kind(kind) {}

bool IntType::needsGenericSubstitutions() const { return false; }

ISizeType::ISizeType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()) {}

bool ISizeType::needsGenericSubstitutions() const { return false; }

NeverType::NeverType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Never, TypeIdentity::empty()) {}

bool NeverType::needsGenericSubstitutions() const { return false; }

UintType::UintType(basic::NodeId id, UintKind kind)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()), kind(kind) {}

bool UintType::needsGenericSubstitutions() const { return false; }

USizeType::USizeType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()) {}

bool USizeType::needsGenericSubstitutions() const { return false; }

StrType::StrType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Str, TypeIdentity::empty()) {}

bool StrType::needsGenericSubstitutions() const { return false; }

TupleType::TupleType(basic::NodeId id, Location loc)
    : BaseType(id, id, TypeKind::Tuple, TypeIdentity::from(loc)) {}

bool TupleType::needsGenericSubstitutions() const { return false; }

TupleType *TupleType::getUnitType(basic::NodeId id) {
  return new TupleType(id, Location::getBuiltinLocation());
}

} // namespace rust_compiler::sema::type_checking::TyTy
