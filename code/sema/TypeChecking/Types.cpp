#include "AST/Enumeration.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "Basic/Ids.h"
#include "Substitutions.h"
#include "TyTy.h"
#include "TypeChecking.h"

#include "../Resolver/Resolver.h"

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

  return resolveSegments(resolvedNodeId, tp->getNodeId(), tp, offset, root);
}

TyTy::BaseType *
TypeResolver::resolveRootPath(std::shared_ptr<ast::types::TypePath> path,
                              size_t *offset, basic::NodeId *resolvedNodeId) {
  assert(false && "to be implemented");

  TyTy::BaseType *rootType = nullptr;
  *offset = 0;

  std::vector<TypePathSegment> segs = path->getSegments();
  NodeId refNodeId = UNKNOWN_NODEID;
  for (unsigned i = 0; i < segs.size(); ++i) {
    bool haveMoreSegments = i != (segs.size() - 1);
    NodeId astNodeId = segs[i].getNodeId();

    if (auto name = resolver->lookupResolvedName(segs[i].getNodeId())) {
      refNodeId = *name;
    } else if (auto type = resolver->lookupResolvedType(segs[i].getNodeId())) {
      refNodeId = *type;
    }

    if (refNodeId == UNKNOWN_NODEID) {
      if (rootType != nullptr and *offset > 0) { // root
        return rootType;
      }
      // report error

      return new TyTy::ErrorType(path->getNodeId());
    }

    // There is no hir

    bool segIsModule = (nullptr != tcx->lookupModule(refNodeId));
    bool segIsCrate = tcx->isCrate(refNodeId);
    if (segIsModule || segIsCrate) {
      if (haveMoreSegments) {
        ++(*offset);
        continue;
      }

      // report error
      return new TyTy::ErrorType(path->getNodeId());
    }

    TyTy::BaseType *lookup = nullptr;
    std::optional<TyTy::BaseType *> result = queryType(astNodeId);
    if (!result) {
      if (*offset == 0) { // root
        // report error
        return new TyTy::ErrorType(path->getNodeId());
      }
      return rootType;
    }

    // enum item?
    if (auto enumItem = tcx->lookupEnumItem(refNodeId)) {
      tcx->insertVariantDefinition(path->getNodeId(),
                                   enumItem->second->getNodeId());
    }

    //if (rootType != nullptr) {
    //  if (lookup->needsGenericSubstitutions()) {
    //    if (!rootType->needsGenericSubstitutions()) {
    //      TyTy::SubstitutionArgumentMappings usedArgs =
    //          TyTy::getUsedSubstitutionArguments(rootType);
    //      lookup = (lookup, usedArgs);
    //      xxx;
    //    }
    //  }
    //}
    //
    //// FIXME
    //
    //if (segs[i].hasGenerics()) {
    //  if (!lookup->hasSubstitutionsDefined()) {
    //    // report error
    //    return new TyTy::ErrorType(path->getNodeId());
    //  }
    //
    //  lookup = xxFn(lookup, segs[i].getGenericArgs());
    //
    //  if (lookup->getKind() == TyTy::TypeKind::Error)
    //    return new TyTy::ErrorType(path->getNodeId());
    //} else if (lookup->needsGenericSubstitutions()) {
    //  lookup = InferStubs(lookup);
    //}

    *resolvedNodeId = refNodeId;
    *offset = *offset + 1;
    rootType = lookup;
  }

  return rootType;
}

TyTy::BaseType *
TypeResolver::resolveSegments(basic::NodeId resolvedNodeId,
                              basic::NodeId pathNodeId,
                              std::shared_ptr<ast::types::TypePath> tp,
                              size_t offset, TyTy::BaseType *pathType) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
