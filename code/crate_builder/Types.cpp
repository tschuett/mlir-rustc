#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "CrateBuilder/CrateBuilder.h"

using namespace rust_compiler::ast::types;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx;

namespace rust_compiler::crate_builder {

mlir::Type CrateBuilder::getType(ast::types::TypeExpression *type) {
  switch (type->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    return getTypeNoBounds(static_cast<TypeNoBounds *>(type));
  }
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be implemented");
  }
  case TypeExpressionKind::TraitObjectType: {
    return getTypeNoBounds(static_cast<TypeNoBounds *>(type));
  }
  }
}

mlir::Type CrateBuilder::getTypeNoBounds(ast::types::TypeNoBounds *noBounds) {
  switch (noBounds->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::TypePath: {
    return getTypePath(static_cast<TypePath *>(noBounds));
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::NeverType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::RawPointerType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::ReferenceType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::ArrayType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::SliceType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::InferredType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::QualifiedPathInType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::BareFunctionType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::MacroInvocation: {
    assert(false && "to be implemented");
  }
  }
  assert(false);
}

mlir::Type CrateBuilder::getTypePath(ast::types::TypePath *path) {
  using TypeKind = rust_compiler::tyctx::TyTy::TypeKind;

  llvm::errs() << "codegen: " << path->getNodeId() << "\n";

  std::optional<TyTy::BaseType *> maybeType =
      tyCtx->lookupType(path->getNodeId());
  if (maybeType) {
    void *type = *maybeType;
    llvm::errs() << "codegen: " << type << "\n";
    llvm::errs() << "codegen: " << ((*maybeType) == nullptr) << "\n";
    llvm::errs() << "codegen: " << (*maybeType)->toString() << "\n";
    switch ((*maybeType)->getKind()) {
    case TypeKind::Bool: {
      assert(false && "to be implemented");
    }
    case TypeKind::Char: {
      assert(false && "to be implemented");
    }
    case TypeKind::Int: {
      assert(false && "to be implemented");
    }
    case TypeKind::Uint: {
      assert(false && "to be implemented");
    }
    case TypeKind::USize: {
      assert(false && "to be implemented");
    }
    case TypeKind::ISize: {
      assert(false && "to be implemented");
    }
    case TypeKind::Float: {
      assert(false && "to be implemented");
    }
    case TypeKind::Closure: {
      assert(false && "to be implemented");
    }
    case TypeKind::Function: {
      assert(false && "to be implemented");
    }
    case TypeKind::Inferred: {
      assert(false && "to be implemented");
    }
    case TypeKind::Never: {
      assert(false && "to be implemented");
    }
    case TypeKind::Str: {
      assert(false && "to be implemented");
    }
    case TypeKind::Tuple: {
      assert(false && "to be implemented");
    }
    case TypeKind::Parameter: {
      assert(false && "to be implemented");
    }
    case TypeKind::Error: {
      assert(false && "to be implemented");
    }
    }
  }
  assert(false);
}

} // namespace rust_compiler::crate_builder
