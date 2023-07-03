#include "ADT/CanonicalPath.h"
#include "AST/GenericArg.h"
#include "AST/GenericArgs.h"
#include "AST/PathIdentSegment.h"
#include "AST/Types/ParenthesizedType.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TraitObjectTypeOneBound.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypePath.h"
#include "Basic/Ids.h"
#include "Resolver.h"

#include <cstdlib>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::basic;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

std::optional<NodeId>
Resolver::resolveType(std::shared_ptr<ast::types::TypeExpression> type,
                      const adt::CanonicalPath &prefix,
                      const adt::CanonicalPath &canonicalPrefix) {
  switch (type->getKind()) {
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be handled later");
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false && "to be handled later");
  }
  case TypeExpressionKind::TypeNoBounds: {
    return resolveTypeNoBounds(std::static_pointer_cast<TypeNoBounds>(type),
                               prefix, canonicalPrefix);
  }
  }
}

std::optional<NodeId> Resolver::resolveTypeNoBounds(
    std::shared_ptr<ast::types::TypeNoBounds> noBounds,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (noBounds->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    return resolveType(
        std::static_pointer_cast<ParenthesizedType>(noBounds)->getType(),
        prefix, canonicalPrefix);
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound:
    return resolveImplTraitTypeOneBound(
        std::static_pointer_cast<ImplTraitTypeOneBound>(noBounds), prefix,
        canonicalPrefix);
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    return resolveTraitObjectTypeOneBound(
        std::static_pointer_cast<TraitObjectTypeOneBound>(noBounds), prefix,
        canonicalPrefix);
  }
  case TypeNoBoundsKind::TypePath: {
    return resolveRelativeTypePath(std::static_pointer_cast<TypePath>(noBounds),
                                   prefix, canonicalPrefix);
  }
  case TypeNoBoundsKind::TupleType: {
    return resolveTupleType(std::static_pointer_cast<TupleType>(noBounds).get(),
                            prefix, canonicalPrefix);
  }
  case TypeNoBoundsKind::NeverType: {
    assert(false && "to be handled later");
  }
  case TypeNoBoundsKind::RawPointerType: {
    return resolveRawPointerType(
        std::static_pointer_cast<RawPointerType>(noBounds), prefix,
        canonicalPrefix);
  }
  case TypeNoBoundsKind::ReferenceType: {
    return resolveReferenceType(
        std::static_pointer_cast<ReferenceType>(noBounds), prefix,
        canonicalPrefix);
  }
  case TypeNoBoundsKind::ArrayType: {
    return resolveArrayType(std::static_pointer_cast<ArrayType>(noBounds),
                            prefix, canonicalPrefix);
  }
  case TypeNoBoundsKind::SliceType: {
    return resolveSliceType(std::static_pointer_cast<SliceType>(noBounds),
                            prefix, canonicalPrefix);
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
    std::shared_ptr<ast::types::TypePath> typePath,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  NodeId moduleScopeId = peekCurrentModuleScope();
  NodeId previousResolveNodeId = moduleScopeId;
  NodeId resolvedNodeId = UNKNOWN_NODEID;

  std::vector<TypePathSegment> segments = typePath->getSegments();

  //  llvm::errs() << "resolveRelativeTypePath: "
  //               << segments[0].getSegment().toString() << "\n";

  // experiment
  //{
  //  if (segments.size() == 1) {
  //    PathIdentSegment ident = segments[0].getSegment();
  //    adt::CanonicalPath path = adt::CanonicalPath::newSegment(
  //        typePath->getNodeId(), Identifier(ident.toString()));
  //    if (auto node = getTypeScope().lookup(path)) {
  //      insertResolvedType(segment.getNodeId(), *node);
  //      resolvedNodeId = *node;
  //      return resolvedNodeId;
  //    }
  //    assert(false);
  //  }
  //}

  // assert(segments.size() == 1 && "to be handled later");

  for (unsigned i = 0; i < segments.size(); ++i) {
    TypePathSegment &segment = segments[i];
    PathIdentSegment ident = segment.getSegment();

    NodeId crateScopeId = peekCrateModuleScope();

    if (segment.hasGenerics())
      resolveGenericArgs(segment.getGenericArgs(), prefix, canonicalPrefix);

    if (segment.hasTypeFunction())
      resolveTypePathFunction(segment.getTypePathFn(), prefix, canonicalPrefix);

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
      // NodeId resolvedNode = UNKNOWN_NODEID;
      //llvm::errs() << "lookups to come for: " << ident.toString() << "\n";
      adt::CanonicalPath path = adt::CanonicalPath::newSegment(
          ident.getNodeId(), Identifier(ident.toString()));
      if (auto node = getTypeScope().lookup(path)) {
        insertResolvedType(segment.getNodeId(), *node);
        resolvedNodeId = *node;
        // llvm::errs() << "it is a type:" << *node << "\n";
      } else if (auto node = getNameScope().lookup(path)) {
        insertResolvedName(segment.getNodeId(), *node);
        resolvedNodeId = *node;
        // llvm::errs() << "it is a name: " << *node << "\n";
      } else if (ident.getKind() == PathIdentSegmentKind::self) {
        moduleScopeId = crateScopeId;
        previousResolveNodeId = moduleScopeId;
        insertResolvedName(segment.getNodeId(), moduleScopeId);
        continue;
      }
    }

    if (resolvedNodeId == UNKNOWN_NODEID &&
        previousResolveNodeId == moduleScopeId) {
      std::optional<adt::CanonicalPath> resolvedChild;
      if (ident.getKind() == PathIdentSegmentKind::Identifier)
        resolvedChild = tyCtx->lookupModuleChild(
            moduleScopeId,
            adt::CanonicalPath::newSegment(ident.getNodeId(),
                                           Identifier(ident.getIdentifier())));
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
          llvm::errs() << llvm::formatv("cannot find path {0} in this scope",
                                        segment.getSegment().toString())
                              .str()
                       << "\n";
          return std::nullopt;
        }
      }
    }

    bool didResolveSegment = resolvedNodeId != UNKNOWN_NODEID;
    //    llvm::errs() << "didResolveSegment:" << didResolveSegment << "\n";
    //    llvm::errs() << "i:" << i << "\n";
    if (didResolveSegment) {
      if (tyCtx->isModule(resolvedNodeId) || tyCtx->isCrate(resolvedNodeId)) {
        moduleScopeId = resolvedNodeId;
      }
      previousResolveNodeId = resolvedNodeId;
    } else if (i == 0) {
      // report error
      llvm::errs() << "print types"
                   << "\n";
      getTypeScope().print();
      llvm::errs() << "print names"
                   << "\n";
      getNameScope().print();
      llvm::errs()
          << llvm::formatv(
                 "{0}: failed to resolve type path {1} in this scope: {2}",
                 segment.getLocation().toString(),
                 segment.getSegment().toString(), resolvedNodeId)
                 .str()
          << "\n";
      llvm::errs() << "Name Resolution pass failed"
                   << "\n";
      exit(EXIT_FAILURE);
      return std::nullopt;
    }
  }

  //  llvm::errs() << "resolve path type: resolved node?"
  //               << "\n";

  if (resolvedNodeId != UNKNOWN_NODEID) {
    //    llvm::errs() << "resolve path type: resolved node"
    //                 << "\n";
    // first name
    if (getNameScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      insertResolvedName(typePath->getNodeId(), resolvedNodeId);
    } else if (getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      insertResolvedType(typePath->getNodeId(), resolvedNodeId);
      //      llvm::errs() << "it is a type"
      //                   << "\n";
    } else {
      llvm_unreachable("");
    }
  }

  return resolvedNodeId;
}

void Resolver::resolveTypePathFunction(
    const ast::types::TypePathFn &fn, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  if (fn.hasInputs()) {
    TypePathFnInputs inpt = fn.getInputs();
    for (auto &type : inpt.getTypes())
      resolveType(type, prefix, canonicalPrefix);
  }

  if (fn.hasType())
    resolveType(fn.getType(), prefix, canonicalPrefix);
}

std::optional<adt::CanonicalPath> Resolver::resolveTypeToCanonicalPath(
    ast::types::TypeExpression *expr, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath result = CanonicalPath::createEmpty();
  switch (expr->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    bool success = resolveTypeNoBoundsToCanonicalPath(
        static_cast<ast::types::TypeNoBounds *>(expr), result, prefix,
        canonicalPrefix);
    if (success)
      return result;
    return std::nullopt;
  }
  case TypeExpressionKind::ImplTraitType: {
    assert(false);
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false);
  }
  }
}

std::optional<basic::NodeId>
Resolver::resolveArrayType(std::shared_ptr<ast::types::ArrayType> arr,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix) {
  resolveExpression(arr->getExpression(), adt::CanonicalPath::createEmpty(),
                    adt::CanonicalPath::createEmpty());
  return resolveType(arr->getType(), prefix, canonicalPrefix);
}

std::optional<basic::NodeId>
Resolver::resolveReferenceType(std::shared_ptr<ast::types::ReferenceType> ref,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {
  return resolveType(ref->getReferencedType(), prefix, canonicalPrefix);
}

std::optional<basic::NodeId> Resolver::resolveTraitObjectTypeOneBound(
    std::shared_ptr<ast::types::TraitObjectTypeOneBound> one,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  return resolveTypeParamBound(one->getBound(), prefix, canonicalPrefix);
}

std::optional<basic::NodeId> Resolver::resolveTypeParamBound(
    std::shared_ptr<ast::types::TypeParamBound> bound,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (bound->getKind()) {
  case TypeParamBoundKind::Lifetime: {
    return std::nullopt;
  }
  case TypeParamBoundKind::TraitBound: {
    return resolveType(std::static_pointer_cast<TraitBound>(bound)->getPath(),
                       prefix, canonicalPrefix);
  }
  }
}

std::optional<basic::NodeId> Resolver::resolveRawPointerType(
    std::shared_ptr<ast::types::RawPointerType> pointer,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  return resolveType(pointer->getType(), prefix, canonicalPrefix);
}

std::optional<basic::NodeId>
Resolver::resolveTupleType(ast::types::TupleType *tuple,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix) {
  for (auto &type : tuple->getTypes())
    resolveType(type, prefix, canonicalPrefix);
  return std::nullopt;
}

std::optional<basic::NodeId>
Resolver::resolveSliceType(std::shared_ptr<ast::types::SliceType> slice,
                           const adt::CanonicalPath &prefix,
                           const adt::CanonicalPath &canonicalPrefix) {
  return resolveType(slice->getType(), prefix, canonicalPrefix);
}

std::optional<basic::NodeId> Resolver::resolveImplTraitTypeOneBound(
    std::shared_ptr<ast::types::ImplTraitTypeOneBound> oneBound,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  std::shared_ptr<ast::types::TypeParamBound> tb = oneBound->getBound();
  switch (tb->getKind()) {
  case TypeParamBoundKind::Lifetime:
    return std::nullopt;
  case TypeParamBoundKind::TraitBound:
    return resolveType(static_pointer_cast<TraitBound>(tb)->getPath(), prefix,
                       canonicalPrefix);
  }
}

bool Resolver::resolveTypeNoBoundsToCanonicalPath(
    ast::types::TypeNoBounds *noBounds, CanonicalPath &result,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
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
    return resolveTypePathToCanonicalPath(
        static_cast<ast::types::TypePath *>(noBounds), result, prefix,
        canonicalPrefix);
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

bool Resolver::resolveTypePathToCanonicalPath(
    ast::types::TypePath *path, CanonicalPath &result,
    const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  NodeId resolvedNode = UNKNOWN_NODEID;
  std::optional<NodeId> aName = tyCtx->lookupResolvedName(path->getNodeId());
  if (aName) {
    resolvedNode = *aName;
  } else {
    std::optional<NodeId> aType = tyCtx->lookupResolvedType(path->getNodeId());
    if (aType)
      resolvedNode = *aType;
  }

  if (resolvedNode == UNKNOWN_NODEID)
    return false;

  std::optional<CanonicalPath> canon = tyCtx->lookupCanonicalPath(resolvedNode);
  if (canon) {
    std::vector<TypePathSegment> segments = path->getSegments();
    auto finalSegment = segments.back();
    if (finalSegment.hasGenerics()) {
      std::vector<CanonicalPath> args;
      GenericArgs generics = finalSegment.getGenericArgs();
      resolveGenericArgs(generics, prefix, canonicalPrefix);
      for (auto &generic : generics.getArgs()) {
        switch (generic.getKind()) {
        case GenericArgKind::Lifetime:
          break;
        case GenericArgKind::Type: {
          std::optional<CanonicalPath> arg = resolveTypeToCanonicalPath(
              generic.getType().get(), prefix, canonicalPrefix);
          if (arg)
            args.push_back(*arg);
          break;
        }
        case GenericArgKind::Const: {
          break;
        }
        case GenericArgKind::Binding: {
          break;
        }
        }
      }

      result = *canon;
      if (!args.empty()) {
        std::string buffer;
        for (size_t i = 0; i < args.size(); ++i) {
          bool hasNext = (i + 1) < args.size();
          const CanonicalPath &arg = args[i];
          buffer += arg.asString();
          if (hasNext)
            buffer += ", ";
        }

        std::string argSegment = "<" + buffer + ">";
        CanonicalPath argumentSegment = CanonicalPath::newSegment(
            finalSegment.getNodeId(), Identifier(argSegment));
        result = result.append(argumentSegment);
      }
    } else {
      result = *canon;
    }
  }
  return true;
}

} // namespace rust_compiler::sema::resolver
