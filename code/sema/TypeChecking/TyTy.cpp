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

CharType::CharType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Char, TypeIdentity::empty()) {}

FloatType::FloatType(basic::NodeId id, FloatKind kind)
    : BaseType(id, id, TypeKind::Float, TypeIdentity::empty()), kind(kind) {}

IntType::IntType(basic::NodeId id, IntKind kind)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()), kind(kind) {}

ISizeType::ISizeType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Int, TypeIdentity::empty()) {}

NeverType::NeverType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Never, TypeIdentity::empty()) {}

UintType::UintType(basic::NodeId id, UintKind kind)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()), kind(kind) {}

USizeType::USizeType(basic::NodeId id)
    : BaseType(id, id, TypeKind::Uint, TypeIdentity::empty()) {}

StrType::StrType(basic::NodeId reference)
    : BaseType(reference, reference, TypeKind::Str, TypeIdentity::empty()) {}

TupleType::TupleType(basic::NodeId id, Location loc)
    : BaseType(id, id, TypeKind::Tuple, TypeIdentity::from(loc)) {}

TupleType *TupleType::getUnitType(basic::NodeId id) {
  return new TupleType(id, Location::getBuiltinLocation());
}

} // namespace rust_compiler::sema::type_checking::TyTy
