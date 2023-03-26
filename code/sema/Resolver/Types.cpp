#include "ADT/CanonicalPath.h"
#include "AST/PathIdentSegment.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "Basic/Ids.h"
#include "Resolver.h"
#include "llvm/Support/ErrorHandling.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::basic;

namespace rust_compiler::sema::resolver {

std::optional<NodeId>
Resolver::resolveType(std::shared_ptr<ast::types::TypeExpression> type) {
  switch (type->getKind()) {
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be handled later");
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false && "to be handled later");
  }
  case TypeExpressionKind::TypeNoBounds: {
    return resolveTypeNoBounds(std::static_pointer_cast<TypeNoBounds>(type));
  }
  }
}

std::optional<NodeId> Resolver::resolveTypeNoBounds(
    std::shared_ptr<ast::types::TypeNoBounds> noBounds) {
  switch (noBounds->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::TypePath: {
    return resolveRelativeTypePath(
        std::static_pointer_cast<TypePath>(noBounds));
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::NeverType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::RawPointerType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ReferenceType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ArrayType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::SliceType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::InferredType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::QualifiedPathInType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::BareFunctionType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::MacroInvocation: {
    assert(false && "to be handled later");
  }
  }
}

/// Note that there is no leading ::
std::optional<NodeId> Resolver::resolveRelativeTypePath(
    std::shared_ptr<ast::types::TypePath> typePath) {

  NodeId moduleScopeId = peekCurrentModuleScope();
  NodeId previousResolveNodeId = moduleScopeId;
  NodeId resolvedNodeId = UNKNOWN_NODEID;

  std::vector<TypePathSegment> segments = typePath->getSegments();

  assert(segments.size() == 1 && "to be handled later");

  for (unsigned i = 0; i < segments.size(); ++i) {
    TypePathSegment &segment = segments[i];
    PathIdentSegment ident = segment.getSegment();

    NodeId crateScopeId = peekCrateModuleScope();

    if (segment.hasGenerics())
      resolveGenericArgs(segment.getGenericArgs());

    if (segment.hasTypeFunction())
      resolveTypePathFunction(segment.getTypePathFn());

    if (i > 0 && ident.getKind() == PathIdentSegmentKind::self) {
      // report error
      llvm::errs() << llvm::formatv("failed to resolve: {0} in path can only "
                                    "used in start position",
                                    segment.getSegment().toString())
                          .str()
                   << "\n";
      return std::nullopt;
    }

    if (ident.getKind() == PathIdentSegmentKind::crate) {
      moduleScopeId = crateScopeId;
      previousResolveNodeId = moduleScopeId;
      insertResolvedName(segment.getNodeId(), moduleScopeId);
      continue;
    }

    if (ident.getKind() == PathIdentSegmentKind::super) {
      if (moduleScopeId == crateScopeId) {
        // report error
        llvm::errs() << "cannot use super at crate scope"
                     << "\n";
        return std::nullopt;
      }
      moduleScopeId = peekParentModuleScope();
      previousResolveNodeId = moduleScopeId;
      insertResolvedName(segment.getNodeId(), moduleScopeId);
      continue;
    }

    if (i == 0) {
      NodeId resolvedNode = UNKNOWN_NODEID;
      adt::CanonicalPath path =
          adt::CanonicalPath::newSegment(segment.getNodeId(), ident.toString());
      if (auto node = getTypeScope().lookup(path)) {
        insertResolvedType(segment.getNodeId(), *node);
        resolvedNodeId = *node;
      } else if (auto node = getNameScope().lookup(path)) {
        insertResolvedName(segment.getNodeId(), *node);
        resolvedNodeId = *node;
      } else if (ident.getKind() == PathIdentSegmentKind::self) {
        moduleScopeId = crateScopeId;
        previousResolveNodeId = moduleScopeId;
        insertResolvedName(segment.getNodeId(), moduleScopeId);
        continue;
      }
    }

    if (resolvedNodeId == UNKNOWN_NODEID &&
        previousResolveNodeId == moduleScopeId) {
      std::optional<adt::CanonicalPath> resolvedChild =
          tyCtx->lookupModuleChild(moduleScopeId, ident.toString());
      if (resolvedChild) {
        NodeId resolvedNode = resolvedChild->getNodeId();
        if (getNameScope().wasDeclDeclaredInCurrentScope(resolvedNode)) {
          resolvedNodeId = resolvedNode;
          insertResolvedName(segment.getNodeId(), resolvedNode);
        } else if (getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNode)) {
          resolvedNodeId = resolvedNode;
          insertResolvedType(segment.getNodeId(), resolvedNode);
        } else {
          // report error
          llvm::errs() << llvm::formatv("cannot file path {0} in this scope",
                                        segment.getSegment().toString())
                              .str()
                       << "\n";
          return std::nullopt;
        }
      }
    }

    bool didResolveSegment = resolvedNodeId != UNKNOWN_NODEID;
    if (didResolveSegment) {
      if (tyCtx->isModule(resolvedNodeId) || tyCtx->isCrate(resolvedNodeId)) {
        moduleScopeId = resolvedNodeId;
      }
      previousResolveNodeId = resolvedNodeId;
    } else if (i == 0) {
      // report error
      llvm::errs() << llvm::formatv(
                          "failed to resolve type path {0} in this scope",
                          segment.getSegment().toString())
                          .str()
                   << "\n";
      return std::nullopt;
    }
  }

  llvm::errs() << "resolve path type: resolved node?"
               << "\n";

  if (resolvedNodeId != UNKNOWN_NODEID) {
    llvm::errs() << "resolve path type: resolved node"
                 << "\n";
    // first name
    if (getNameScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      insertResolvedName(typePath->getNodeId(), resolvedNodeId);
    } else if (getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      insertResolvedType(typePath->getNodeId(), resolvedNodeId);
    } else {
      llvm_unreachable("");
    }
  }

  return resolvedNodeId;
}

void Resolver::resolveTypePathFunction(const ast::types::TypePathFn &) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
