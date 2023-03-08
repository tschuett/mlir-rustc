#include "ADT/CanonicalPath.h"
#include "Resolver.h"
#include "Sema/Mappings.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;
using namespace rust_compiler::basic;
using namespace rust_compiler::sema;

namespace rust_compiler::sema::resolver {

void Resolver::resolveVisItemNoRecurse(
    std::shared_ptr<ast::VisItem> visItem, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {

  NodeId currentModule = peekCurrentModuleScope();
  Mappings::get()->insertChildItemToParentModuleMapping(visItem->getNodeId(),
                                                        currentModule);
  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    break;
  }
  case VisItemKind::ExternCrate: {
    break;
  }
  case VisItemKind::UseDeclaration: {
    break;
  }
  case VisItemKind::Function: {
    resolveFunctionNoRecurse(std::static_pointer_cast<Function>(visItem),
                             prefix, canonicalPrefix);
    break;
  }
  case VisItemKind::TypeAlias: {
    break;
  }
  case VisItemKind::Struct: {
    break;
  }
  case VisItemKind::Enumeration: {
    break;
  }
  case VisItemKind::Union: {
    break;
  }
  case VisItemKind::ConstantItem: {
    break;
  }
  case VisItemKind::StaticItem: {
    break;
  }
  case VisItemKind::Trait: {
    break;
  }
  case VisItemKind::Implementation: {
    break;
  }
  case VisItemKind::ExternBlock: {
    break;
  }
  }
}

void Resolver::resolveFunctionNoRecurse(
    std::shared_ptr<ast::Function> fun, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  CanonicalPath segment =
      CanonicalPath::newSegment(fun->getNodeId(), fun->getName());
  CanonicalPath path = canonicalPrefix.append(segment);

  /// FIXME

  NodeId currentModule = peekCurrentModuleScope();
  Mappings::get()->insertModuleChildItem(currentModule, segment);
  Mappings::get()->insertCanonicalPath(fun->getNodeId(), path);
}

void Resolver::resolveMacroItemNoRecurse(
    std::shared_ptr<ast::MacroItem>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {}

} // namespace rust_compiler::sema::resolver
