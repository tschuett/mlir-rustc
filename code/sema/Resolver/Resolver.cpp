#include "Resolver.h"

#include "ADT/CanonicalPath.h"
#include "AST/ConstantItem.h"
#include "AST/Implementation.h"
#include "AST/StaticItem.h"
#include "Basic/Ids.h"

#include <memory>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

Resolver::Resolver() noexcept
    : mappings(Mappings::get()), tyctx(TypeCheckContext::get()),
      nameScope(Scope(mappings->getCurrentCrate())),
      typeScope(Scope(mappings->getCurrentCrate())),
      labelScope(Scope(mappings->getCurrentCrate())),
      macroScope(Scope(mappings->getCurrentCrate())),
      globalTypeNodeId(UNKNOWN_NODEID), unitTyNodeId(UNKNOWN_NODEID) {
  generateBuiltins();
}

void Resolver::resolveCrate(std::shared_ptr<ast::Crate> crate) {
  // FIXME

  // setup scopes

  NodeId scopeNodeId = crate->getNodeId();
  getNameScope().push(scopeNodeId);
  getTypeScope().push(scopeNodeId);
  getLabelScope().push(scopeNodeId);
  pushNewNameRib(getNameScope().peek());
  pushNewTypeRib(getTypeScope().peek());
  pushNewLabelRib(getLabelScope().peek());

  // get the root segment
  NodeId crateId = crate->getNodeId();
  CanonicalPath cratePrefix =
      CanonicalPath::newSegment(crateId, crate->getCrateName());
  cratePrefix.setCrateNum(crate->getCrateNum());

  pushNewModuleScope(crateId);

  // only gather top-level
  for (auto &item : crate->getItems()) {
    switch (item->getItemKind()) {
    case ItemKind::VisItem: {
      resolveVisItemNoRecurse(std::static_pointer_cast<VisItem>(item),
                              CanonicalPath::createEmpty(), cratePrefix);
      break;
    }
    case ItemKind::MacroItem: {
      resolveMacroItemNoRecurse(std::static_pointer_cast<MacroItem>(item),
                                CanonicalPath::createEmpty(), cratePrefix);
      break;
    }
    }
  }

  // recursive
  for (auto &item : crate->getItems()) {
    switch (item->getItemKind()) {
    case ItemKind::VisItem: {
      resolveVisItem(std::static_pointer_cast<VisItem>(item),
                     CanonicalPath::createEmpty(), cratePrefix);
      break;
    }
    case ItemKind::MacroItem: {
      resolveMacroItem(std::static_pointer_cast<MacroItem>(item),
                       CanonicalPath::createEmpty(), cratePrefix);
      break;
    }
    }
  }

  // done
  popModuleScope();
}

void Resolver::resolveVisItem(std::shared_ptr<ast::VisItem> visItem,
                              const adt::CanonicalPath &prefix,
                              const adt::CanonicalPath &canonicalPrefix) {
  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    std::shared_ptr<ast::Module> mod =
        std::static_pointer_cast<Module>(visItem);
    basic::NodeId modNodeId = mod->getNodeId();
    Mappings::get()->insertModule(mod.get());
    pushNewModuleScope(modNodeId);
    resolveModule(mod, prefix, canonicalPrefix);
    popModuleScope();
    break;
  }
  case VisItemKind::ExternCrate: {
    break;
  }
  case VisItemKind::UseDeclaration: {
    break;
  }
  case VisItemKind::Function: {
    resolveFunction(std::static_pointer_cast<Function>(visItem), prefix,
                    canonicalPrefix);
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
    resolveConstantItem(std::static_pointer_cast<ConstantItem>(visItem), prefix,
                        canonicalPrefix);
    break;
  }
  case VisItemKind::StaticItem: {
    resolveStaticItem(std::static_pointer_cast<StaticItem>(visItem), prefix,
                      canonicalPrefix);
    break;
  }
  case VisItemKind::Trait: {
    break;
  }
  case VisItemKind::Implementation: {
    resolveImplementation(
        std::static_pointer_cast<ast::Implementation>(visItem), prefix,
        canonicalPrefix);
    break;
  }
  case VisItemKind::ExternBlock: {
    break;
  }
  }
}

void Resolver::resolveImplementation(
    std::shared_ptr<ast::Implementation> impl, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    resolveInherentImpl(std::static_pointer_cast<ast::InherentImpl>(impl),
                        prefix, canonicalPrefix);
    break;
  }
  case ImplementationKind::TraitImpl: {
    resolveTraitImpl(std::static_pointer_cast<ast::TraitImpl>(impl), prefix,
                     canonicalPrefix);
    break;
  }
  }
}

void Resolver::resolveMacroItem(std::shared_ptr<ast::MacroItem>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {}

void Resolver::resolveInherentImpl(std::shared_ptr<ast::InherentImpl>,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {}

void Resolver::resolveTraitImpl(std::shared_ptr<ast::TraitImpl>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {}

void Resolver::resolveVisibility(std::optional<ast::Visibility> vis) {
  // FIXME
}

} // namespace rust_compiler::sema::resolver

// items: canoncialPath

// namespaces: types, macros, values, lifeftimes
