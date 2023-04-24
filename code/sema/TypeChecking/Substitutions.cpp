#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *TypeResolver::applySubstitutions(TyTy::BaseType *type, Location,
                                                 const GenericArgs &) {
  assert(false);

  switch (type->getKind()) {
  case TypeKind::Bool:
  case TypeKind::Char:
  case TypeKind::Int:
  case TypeKind::Uint:
  case TypeKind::USize:
  case TypeKind::ISize:
  case TypeKind::Float:
  case TypeKind::Closure:
  case TypeKind::Inferred:
  case TypeKind::Never:
  case TypeKind::Error:
  case TypeKind::Str:
  case TypeKind::Tuple:
  case TypeKind::Parameter:
  case TypeKind::StructField:
    return type;
  case TypeKind::ADT: {
    assert(false);
  }
  case TypeKind::Function: {
    assert(false);
  }
  }
}

} // namespace rust_compiler::sema::type_checking
