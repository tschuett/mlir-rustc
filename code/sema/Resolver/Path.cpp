#include "ADT/CanonicalPath.h"
#include "AST/SimplePathSegment.h"
#include "Basic/Ids.h"
#include "Lexer/KeyWords.h"
#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <optional>

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;
using namespace rust_compiler::basic;
using namespace rust_compiler::lexer;

namespace rust_compiler::sema::resolver {

std::optional<NodeId> Resolver::resolveSimplePath(const ast::SimplePath &path) {

  NodeId crateScopeId = peekCrateModuleScope();
  NodeId moduleScopeId = peekCurrentModuleScope();

  NodeId resolvedNodeId = UNKNOWN_NODEID;

  for (size_t i = 0; i < path.getNrOfSegments(); i++) {
    SimplePathSegment segment = path.getSegment(i);

    resolvedNodeId = UNKNOWN_NODEID;

    if (segment.isKeyWord()) {
      KeyWordKind keyWord = segment.getKeyWord();

      if (keyWord == KeyWordKind::KW_CRATE) {
        moduleScopeId = crateScopeId;
        insertResolvedName(segment.getNodeId(), moduleScopeId);

        continue;
      } else if (keyWord == KeyWordKind::KW_SUPER) {
        if (moduleScopeId == crateScopeId) {
          // report error
          llvm::errs() << "cannot use super at the crate scope"
                       << "\n";
          return std::nullopt;
        }
        moduleScopeId = peekParentModuleScope();
        insertResolvedName(segment.getNodeId(), moduleScopeId);

        continue;
      }
    }

    std::optional<adt::CanonicalPath> resolvedChild =
        mappings->lookupModuleChild(moduleScopeId, segment.getName());
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
        llvm::errs() << "cannot find path in this scope; " << segment.asString()
                     << "\n";
        return std::nullopt;
      }
    }

    if (resolvedNodeId == UNKNOWN_NODEID && i == 0) {
      NodeId resolveNode = UNKNOWN_NODEID;
      CanonicalPath p =
          CanonicalPath::newSegment(segment.getNodeId(), segment.getName());
      std::optional<NodeId> node = getNameScope().lookup(p);
      if (node) {
        resolvedNodeId = *node;
        insertResolvedName(segment.getNodeId(), *node);
      } else {
        std::optional<NodeId> node = getTypeScope().lookup(p);
        if (node) {
          resolvedNodeId = *node;
          insertResolvedType(segment.getNodeId(), *node);
        }
      }
    }

    if (resolvedNodeId == UNKNOWN_NODEID) {
      llvm::errs() << "cannot find simple path segment in current scope: "
                   << segment.asString() << "\n";
      return std::nullopt;
    }

    if (mappings->isModule(resolvedNodeId))
      moduleScopeId = resolvedNodeId;
  }

  if (resolvedNodeId != UNKNOWN_NODEID) {

    if (getNameScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      // name scope
      insertResolvedName(path.getNodeId(), resolvedNodeId);
    } else if (getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
      // type scope
      insertResolvedType(path.getNodeId(), resolvedNodeId);
    } else {
      // unreachable
    }
  }

  return resolvedNodeId;
}

} // namespace rust_compiler::sema::resolver
