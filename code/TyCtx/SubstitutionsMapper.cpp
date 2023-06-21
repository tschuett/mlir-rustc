#include "TyCtx/SubstitutionsMapper.h"

#include "TyCtx/Substitutions.h"
#include "TyCtx/TyTy.h"

using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::tyctx {

TyTy::BaseType *InternalSubstitutionsMapper::resolve(
    TyTy::BaseType *base, TyTy::SubstitutionArgumentMappings &mappings) {
  switch (base->getKind()) {
  case TypeKind::Bool: {
    assert(false);
  }
  case TypeKind::Char: {
    assert(false);
  }
  case TypeKind::Int: {
    assert(false);
  }
  case TypeKind::Uint: {
    assert(false);
  }
  case TypeKind::USize: {
    assert(false);
  }
  case TypeKind::ISize: {
    assert(false);
  }
  case TypeKind::Float: {
    assert(false);
  }
  case TypeKind::Closure: {
    assert(false);
  }
  case TypeKind::Function: {
    assert(false);
  }
  case TypeKind::Inferred: {
    assert(false);
  }
  case TypeKind::Never: {
    assert(false);
  }
  case TypeKind::Str: {
    assert(false);
  }
  case TypeKind::Tuple: {
    assert(false);
  }
  case TypeKind::Parameter: {
    assert(false);
  }
  case TypeKind::ADT: {
    assert(false);
  }
  case TypeKind::Array: {
    assert(false);
  }
  case TypeKind::Slice: {
    assert(false);
  }
  case TypeKind::Projection: {
    assert(false);
  }
  case TypeKind::Dynamic: {
    assert(false);
  }
  case TypeKind::PlaceHolder: {
    assert(false);
  }
  case TypeKind::FunctionPointer: {
    assert(false);
  }
  case TypeKind::RawPointer: {
    assert(false);
  }
  case TypeKind::Reference: {
    assert(false);
  }
  case TypeKind::Error: {
    assert(false);
  }
  }
  assert(false);
}

TyTy::BaseType *
SubstitutionsMapper::resolve(TyTy::BaseType *base, Location loc,
                             sema::type_checking::TypeResolver *resolver,
                             ast::GenericArgs *generics) {
  this->generics = generics;
  this->loc = loc;

  switch (base->getKind()) {
  case TypeKind::Bool: {
    assert(false);
  }
  case TypeKind::Char: {
    assert(false);
  }
  case TypeKind::Int: {
    assert(false);
  }
  case TypeKind::Uint: {
    assert(false);
  }
  case TypeKind::USize: {
    assert(false);
  }
  case TypeKind::ISize: {
    assert(false);
  }
  case TypeKind::Float: {
    assert(false);
  }
  case TypeKind::Closure: {
    assert(false);
  }
  case TypeKind::Function: {
    assert(false);
  }
  case TypeKind::Inferred: {
    assert(false);
  }
  case TypeKind::Never: {
    assert(false);
  }
  case TypeKind::Str: {
    assert(false);
  }
  case TypeKind::Tuple: {
    assert(false);
  }
  case TypeKind::Parameter: {
    assert(false);
  }
  case TypeKind::ADT: {
    TyTy::ADTType *concrete = nullptr;
    // TyTy::ADTType concrete = static_cast<TyTy::ADTType*>(base);
    if (!generics) {
      TyTy::BaseType *substs =
          static_cast<TyTy::ADTType *>(base)->inferSubstitions(loc);
      assert(substs->getKind() == TypeKind::ADT);
      return static_cast<TyTy::ADTType *>(substs);
    } else { //  generics
      TyTy::SubstitutionArgumentMappings mappings =
          static_cast<TyTy::ADTType *>(base)->getMappingsFromGenericArgs(
              *generics, resolver);
      if (mappings.isError())
        assert(false);
      concrete =
          static_cast<TyTy::ADTType *>(base)->handleSubstitions(mappings);
    }
    if (concrete != nullptr)
      return concrete;
    assert(false);
  }
  case TypeKind::Array: {
    assert(false);
  }
  case TypeKind::Slice: {
    assert(false);
  }
  case TypeKind::Projection: {
    assert(false);
  }
  case TypeKind::Dynamic: {
    assert(false);
  }
  case TypeKind::PlaceHolder: {
    assert(false);
  }
  case TypeKind::FunctionPointer: {
    assert(false);
  }
  case TypeKind::RawPointer: {
    assert(false);
  }
  case TypeKind::Reference: {
    assert(false);
  }
  case TypeKind::Error: {
    assert(false);
  }
  }
  assert(false);
}

TyTy::BaseType *
SubstitutionsMapper::infer(TyTy::BaseType *base, Location loc,
                           sema::type_checking::TypeResolver *resolver) {
  return resolve(base, loc, resolver, nullptr);
}

} // namespace rust_compiler::tyctx
