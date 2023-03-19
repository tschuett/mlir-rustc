#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkType(std::shared_ptr<ast::types::TypeExpression> te) {
  switch (te->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    return checkTypeNoBounds(std::static_pointer_cast<TypeNoBounds>(te));
  }
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be implemented");
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false && "to be implemented");
  }
  }
}

void TypeResolver::checkWhereClause(const ast::WhereClause &) {
  assert(false && "to be implemented");
}

void TypeResolver::checkGenericParams(
    const GenericParams &, std::vector<TyTy::SubstitutionParamMapping> &) {
  assert(false && "to be implemented");
}

TyTy::BaseType *
TypeResolver::checkTypeNoBounds(std::shared_ptr<ast::types::TypeNoBounds> no) {
  switch (no->getKind()) {
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
    return checkTypePath(std::static_pointer_cast<TypePath>(no));
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
}

TyTy::BaseType *
TypeResolver::checkTypePath(std::shared_ptr<ast::types::TypePath> tp) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
