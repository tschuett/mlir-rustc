#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *TypeResolver::applyGenericArgs(TyTy::BaseType *type,
                                               Location loc,
                                               const GenericArgs &args) {
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
  case TypeKind::Array:
  case TypeKind::RawPointer:
  case TypeKind::Slice:
  case TypeKind::Dynamic:
  case TypeKind::FunctionPointer:
  case TypeKind::Reference:
    return type;
  case TypeKind::ADT: {
    return applyGenericArgsToADT(static_cast<TyTy::ADTType *>(type), loc, args);
  }
  case TypeKind::Projection: {
    assert(false);
  }
  case TypeKind::Function: {
    assert(false);
  }
  case TypeKind::PlaceHolder: {
    assert(false);
  }
  }
}

TyTy::BaseType *TypeResolver::applyGenericArgsToADT(TyTy::ADTType *type,
                                                    Location loc,
                                                    const GenericArgs &args) {
  if (args.getNumberOfArgs() == 0) {
    TyTy::BaseType *substs = type->inferSubstitutions(loc);
    assert(substs->getKind() == TypeKind::ADT);
    return static_cast<TyTy::ADTType *>(substs);
  } else {
    TyTy::SubstitutionArgumentMappings mappings =
        type->getMappingsFromGenericArgs(args);
    // if (mappings.isError())
    //   return
    return type->handleSubstitutions(mappings);
  }
}

TyTy::BaseType *TypeResolver::applySubstitutionMappings(
    TyTy::BaseType *, const TyTy::SubstitutionArgumentMappings &) {
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
