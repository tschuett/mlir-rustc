#include "AST/GenericParam.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"
#include "Unification.h"

using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

bool TypeResolver::checkGenericParamsAndArgs(const TyTy::BaseType *type,
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
    const ADTType *adt = static_cast<const ADTType *>(type);
    const GenericParameters *gp = static_cast<const GenericParameters *>(adt);
    if (gp->needsSubstitution())
      return checkGenericParamsAndArgs(*gp->getGenericParams(), args);
    return true;
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

bool TypeResolver::checkGenericParamsAndArgs(const ast::GenericParams &params,
                                             const ast::GenericArgs &args) {
  std::vector<GenericParam> gps = params.getGenericParams();
  std::vector<GenericArg> gas = args.getArgs();

  for (unsigned i = 0; i < gps.size(); ++i) {
    switch (gas[i].getKind()) {
    case GenericArgKind::Lifetime: {
      switch (gps[i].getKind()) {
      case ast::GenericParamKind::LifetimeParam: {
        break;
      }
      case ast::GenericParamKind::TypeParam: {
        // report error
        break;
      }
      case ast::GenericParamKind::ConstParam: {
        // report error
        break;
      }
      }
      break;
    }
    case GenericArgKind::Type: {
      switch (gps[i].getKind()) {
      case ast::GenericParamKind::LifetimeParam: {
        // report error
        break;
      }
      case ast::GenericParamKind::TypeParam: {
        TypeParam tp = gps[i].getTypeParam();
        checkType(gas[i].getType());
        if (tp.hasType()) {
          // unifyWithSite();
        }
        break;
      }
      case ast::GenericParamKind::ConstParam: {
        // report error
        break;
      }
      }
      break;
    }
    case GenericArgKind::Const: {
      switch (gps[i].getKind()) {
      case ast::GenericParamKind::LifetimeParam: {
        // report error
        break;
      }
      case ast::GenericParamKind::TypeParam: {
        // report error
        break;
      }
      case ast::GenericParamKind::ConstParam: {
        // TODO type matches expression
        break;
      }
      }
      break;
    }
    case GenericArgKind::Binding: {
      switch (gps[i].getKind()) {
      case ast::GenericParamKind::LifetimeParam: {
        // report error
        break;
      }
      case ast::GenericParamKind::TypeParam: {
        // Binding: Identifier = Type
        break;
      }
      case ast::GenericParamKind::ConstParam: {
        // report error
        break;
      }
      }
      break;
    }
    }
  }
  return false;
}

} // namespace rust_compiler::sema::type_checking
