#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "CrateBuilder/CrateBuilder.h"
#include "TyCtx/TyTy.h"

#include <mlir/IR/BuiltinTypes.h>

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

mlir::Type CrateBuilder::getExpression(ast::Expression *expr) {
  using namespace rust_compiler::tyctx::TyTy;

  std::optional<TyTy::BaseType *> maybeType =
      tyCtx->lookupType(expr->getNodeId());
  if (maybeType) {
    return convertTyTyToMLIR(*maybeType);
  }
  assert(false);
}

mlir::Type CrateBuilder::convertTyTyToMLIR(TyTy::BaseType *type) {
  using TypeKind = rust_compiler::tyctx::TyTy::TypeKind;
  using IntKind = rust_compiler::tyctx::TyTy::IntKind;
  using UintKind = rust_compiler::tyctx::TyTy::UintKind;

  switch (type->getKind()) {
  case TypeKind::Bool: {
    assert(false && "to be implemented");
  }
  case TypeKind::Char: {
    assert(false && "to be implemented");
  }
  case TypeKind::Int: {
    switch (static_cast<TyTy::IntType *>(type)->getIntKind()) {
    case IntKind::I8: {
      assert(false && "to be implemented");
    }
    case IntKind::I16: {
      assert(false && "to be implemented");
    }
    case IntKind::I32: {
      return builder.getI32Type();
    }
    case IntKind::I64: {
      assert(false && "to be implemented");
    }
    case IntKind::I128: {
      assert(false && "to be implemented");
    }
    }
  }
  case TypeKind::Uint: {
    switch (static_cast<TyTy::UintType *>(type)->getUintKind()) {
    case UintKind::U8: {
      assert(false);
    }
    case UintKind::U16: {
      assert(false);
    }
    case UintKind::U32: {
      return builder.getIntegerType(32, false);
    }
    case UintKind::U64: {
      assert(false);
    }
    case UintKind::U128: {
      assert(false);
    }
    }
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
  case TypeKind::ADT: {
    assert(false && "to be implemented");
  }
  case TypeKind::Projection: {
    assert(false && "to be implemented");
  }
  case TypeKind::Array: {
    assert(false && "to be implemented");
  }
  case TypeKind::Error: {
    assert(false && "to be implemented");
  }
  case TypeKind::Dynamic: {
    assert(false && "to be implemented");
  }
  case TypeKind::Slice: {
    assert(false && "to be implemented");
  }
  case TypeKind::PlaceHolder: {
    assert(false && "to be implemented");
  }
  case TypeKind::FunctionPointer: {
    assert(false && "to be implemented");
  }
  case TypeKind::RawPointer: {
    assert(false && "to be implemented");
  }
  case TypeKind::Reference: {
    assert(false && "to be implemented");
  }
  }
}

mlir::Type CrateBuilder::getTypePath(ast::types::TypePath *path) {
  using namespace rust_compiler::tyctx::TyTy;

  std::optional<TyTy::BaseType *> maybeType =
      tyCtx->lookupType(path->getNodeId());
  if (maybeType) {
    return convertTyTyToMLIR(*maybeType);
  }
  assert(false);
}

mlir::MemRefType CrateBuilder::getMemRefType(TyTy::BaseType *base) {
  using namespace rust_compiler::tyctx::TyTy;

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
    TyTy::ArrayType *array = static_cast<TyTy::ArrayType *>(base);
    mlir::Type elementType = convertTyTyToMLIR(array->getElementType());
    uint64_t length =
        foldAsUsizeExpression(array->getCapacityExpression().get());
    return mlir::MemRefType::Builder(length, elementType);
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
}

} // namespace rust_compiler::crate_builder
