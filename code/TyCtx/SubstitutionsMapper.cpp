#include "TyCtx/SubstitutionsMapper.h"

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
                             ast::GenericArgs *generics) {
  assert(false);
}

} // namespace rust_compiler::tyctx
