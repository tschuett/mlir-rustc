#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "Basic/Ids.h"
#include "TyTy.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::basic;

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
  assert(!tp->hasLeadingPathSep() && "to be implemented");

  size_t offset = 0;
  NodeId resolvedNodeId = UNKNOWN_NODEID;
  TyTy::BaseType *root = resolveRootPath(tp, &offset, &resolvedNodeId);
  if (root->getKind() == TyTy::TypeKind::Error)
    return nullptr;

  root->setReference(tp->getNodeId());
  tcx->insertImplicitType(tp->getNodeId(), root);

  if (offset >= tp->getSegments().size())
    return root;

  resolveSegments(resolvedNodeId, tp->getNodeId(), tp, offset, root);
}

TyTy::BaseType *
TypeResolver::resolveRootPath(std::shared_ptr<ast::types::TypePath> path,
                              size_t *offset, basic::NodeId *resolvedNodeId) {
  assert(false && "to be implemented");
}

TyTy::BaseType *
TypeResolver::resolveSegments(basic::NodeId resolvedNodeId,
                              basic::NodeId pathNodeId,
                              std::shared_ptr<ast::types::TypePath> tp,
                              size_t offset, TyTy::BaseType *pathType) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
