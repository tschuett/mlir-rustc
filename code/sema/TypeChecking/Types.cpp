#include "AST/Enumeration.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "Basic/Ids.h"
#include "PathProbing.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"
#include <llvm/Support/raw_ostream.h>

#include "../Resolver/Resolver.h"

#include <memory>
#include <optional>
#include <tuple>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::basic;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkType(std::shared_ptr<ast::types::TypeExpression> te) {
  std::optional<TyTy::BaseType *> type = tcx->lookupType(te->getNodeId());
  if (type)
    return *type;

  TyTy::BaseType *result = nullptr;
  switch (te->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    result = checkTypeNoBounds(std::static_pointer_cast<TypeNoBounds>(te));
    break;
  }
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be implemented");
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false && "to be implemented");
  }
  }
  assert(result);
  assert(result->getKind() != TyTy::TypeKind::Error);
  tcx->insertType(te->getIdentity(), result);
  return result;
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
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::NeverType: {
    return checkNeverType(std::static_pointer_cast<ast::types::NeverType>(no));
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
  assert(false && "to be implemented");
}

TyTy::BaseType *
TypeResolver::checkTypePath(std::shared_ptr<ast::types::TypePath> tp) {
  assert(!tp->hasLeadingPathSep() && "to be implemented");

  size_t offset = 0;
  NodeId resolvedNodeId = UNKNOWN_NODEID;
  TyTy::BaseType *root = resolveRootPathType(tp, &offset, &resolvedNodeId);
  if (root->getKind() == TyTy::TypeKind::Error) {
    assert(false);
    return new TyTy::ErrorType(tp->getNodeId());
  }

  root->setReference(tp->getNodeId());
  tcx->insertImplicitType(tp->getNodeId(), root);

  if (offset >= tp->getSegments().size())
    return root;

  TyTy::BaseType *segments =
      resolveSegmentsType(resolvedNodeId, tp->getNodeId(), tp, offset, root);
  assert(segments);
  assert(segments->getKind() != TyTy::TypeKind::Error);
  return segments;
}

TyTy::BaseType *
TypeResolver::resolveRootPathType(std::shared_ptr<ast::types::TypePath> path,
                                  size_t *offset,
                                  basic::NodeId *resolvedNodeId) {
  // assert(false && "to be implemented");

  TyTy::BaseType *rootType = nullptr;
  *offset = 0;

  std::vector<TypePathSegment> segs = path->getSegments();

  if (segs.size() == 1)
    if (auto t = tcx->lookupBuiltin(segs[0].getSegment().toString())) {
      *offset = 1;
//      llvm::errs() << path->getNodeId() << " -> " << t->toString() << "\n";
//      llvm::errs() << (void*)t << "\n";
      return t;
    }

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
      if (*offset == 0) { // root
        // report error
        llvm::errs() << "unknown reference for resolved name: "
                     << segs[i].getSegment().toString() << " @ "
                     << segs[i].getLocation().toString() << "\n";
        return new TyTy::ErrorType(path->getNodeId());
      }
      return rootType;
    }

    // There is no hir

    bool segIsModule = nullptr != tcx->lookupModule(refNodeId);
    bool segIsCrate = tcx->isCrate(refNodeId);
    if (segIsModule || segIsCrate) {
      if (haveMoreSegments) {
        ++(*offset);
        continue;
      }

      // report error
      llvm::errs() << "expected value, but found crate or module "
                   << "\n";
      return new TyTy::ErrorType(path->getNodeId());
    }

    TyTy::BaseType *lookup = nullptr;
    std::optional<TyTy::BaseType *> result = queryType(astNodeId);
    if (!result) {
      if (*offset == 0) { // root
        // report error
        llvm::errs() << "queryType failed: failed to resolve root segment"
                     << "\n";
        return new TyTy::ErrorType(path->getNodeId());
      }
      return rootType;
    }

    //    // enum item?
    //    if (auto enumItem = tcx->lookupEnumItem(refNodeId)) {
    //      tcx->insertVariantDefinition(path->getNodeId(),
    //                                   enumItem->second->getNodeId());
    //    }

    // if (rootType != nullptr) {
    //   if (lookup->needsGenericSubstitutions()) {
    //     if (!rootType->needsGenericSubstitutions()) {
    //       TyTy::SubstitutionArgumentMappings usedArgs =
    //           TyTy::getUsedSubstitutionArguments(rootType);
    //       lookup = (lookup, usedArgs);
    //       xxx;
    //     }
    //   }
    // }
    //
    //// FIXME
    //
    // if (segs[i].hasGenerics()) {
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
TypeResolver::resolveSegmentsType(basic::NodeId rootResolvedNodeId,
                                  basic::NodeId pathNodeId,
                                  std::shared_ptr<ast::types::TypePath> tp,
                                  size_t offset, TyTy::BaseType *pathType) {
  assert(false && "to be implemented");

  // basic::NodeId resolvedNodeId = rootResolvedNodeId;
  TyTy::BaseType *prevSegment = pathType;

  std::vector<TypePathSegment> segs = tp->getSegments();

  for (unsigned i = offset; i < segs.size(); ++i) {
    segs[i];

    bool receiverIsGeneric =
        prevSegment->getKind() == TyTy::TypeKind::Parameter;
    bool probeBounds = true;
    bool probeImpls = !receiverIsGeneric;
    bool ignoreTraitItems = !receiverIsGeneric;

    std::set<PathProbeCandidate> candidates = probeTypePath(
        prevSegment, segs[i].getSegment(), probeImpls, false, ignoreTraitItems);

    if (candidates.size() == 0) {
      candidates = probeTypePath(prevSegment, segs[i].getSegment(), false,
                                 probeBounds, ignoreTraitItems);

      if (candidates.size() == 0) {
        // report error
        return new TyTy::ErrorType(tp->getNodeId());
      }
    }

    if (candidates.size() > 1) {
      // report error
      return new TyTy::ErrorType(tp->getNodeId());
    }

    // FIXME
  }
}

TyTy::BaseType *
TypeResolver::checkNeverType(std::shared_ptr<ast::types::NeverType>) {
  std::optional<TyTy::BaseType *> never = tcx->lookupBuiltin("!");
  if (never)
    return *never;
  return nullptr;
}

} // namespace rust_compiler::sema::type_checking
