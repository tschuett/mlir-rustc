#include "AST/Types/ImplTraitType.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TraitObjectType.h"
#include "Resolver.h"

using namespace rust_compiler::ast::types;

namespace rust_compiler::sema::resolver {

std::string Resolver::resolveTypeToString(ast::types::TypeExpression *typ) {
  switch (typ->getKind()) {
  case TypeExpressionKind::TypeNoBounds:
    return resolveTypeNoBoundsToString(static_cast<TypeNoBounds *>(typ));
  case TypeExpressionKind::ImplTraitType:
    return resolveImplTraitTypeToString(static_cast<ImplTraitType *>(typ));
  case TypeExpressionKind::TraitObjectType:
    return resolveTraitObjectTypeToString(static_cast<TraitObjectType *>(typ));
  }
}

std::string
Resolver::resolveTypeNoBoundsToString(ast::types::TypeNoBounds *noBounds) {
  assert(false);
  switch (noBounds->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    assert(false);
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false);
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound: {
    assert(false);
  }
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    assert(false);
  }
  case TypeNoBoundsKind::TypePath: {
    assert(false);
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false);
  }
  case TypeNoBoundsKind::NeverType: {
    assert(false);
  }
  case TypeNoBoundsKind::RawPointerType: {
    assert(false);
  }
  case TypeNoBoundsKind::ReferenceType: {
    assert(false);
  }
  case TypeNoBoundsKind::ArrayType: {
    assert(false);
  }
  case TypeNoBoundsKind::SliceType: {
    assert(false);
  }
  case TypeNoBoundsKind::InferredType: {
    assert(false);
  }
  case TypeNoBoundsKind::QualifiedPathInType: {
    assert(false);
  }
  case TypeNoBoundsKind::BareFunctionType: {
    assert(false);
  }
  case TypeNoBoundsKind::MacroInvocation: {
    assert(false);
  }
  }
}

std::string
Resolver::resolveImplTraitTypeToString(ast::types::ImplTraitType *) {
  assert(false);
}

std::string
Resolver::resolveTraitObjectTypeToString(ast::types::TraitObjectType *) {
  assert(false);
}

} // namespace rust_compiler::sema::resolver
