#include "Resolver.h"

#include "ADT/CanonicalPath.h"
#include "AST/ConstantItem.h"
#include "AST/Implementation.h"
#include "AST/StaticItem.h"
#include "Basic/Ids.h"

#include <memory>
#include <optional>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

void Rib::insertName(const adt::CanonicalPath &path, basic::NodeId id,
                     Location loc, bool shadow, RibKind kind) {
  auto it = pathMappings.find(path);
  if (it != pathMappings.end() && !shadow)
    return;

  pathMappings[path] = id;
  reversePathMappings.insert({id, path});
  declsWithinRib.insert({id, loc});
  references[id] = {};
  declTypeMappings.insert({id, kind});
}

std::optional<basic::NodeId> Rib::lookupName(const adt::CanonicalPath &ident) {
  auto it = pathMappings.find(ident);
  if (it == pathMappings.end())
    return std::nullopt;

  return it->second;
}

Rib *Scope::peek() { return stack.back(); }

void Scope::push(NodeId id) { stack.push_back(new Rib(getCrateNum(), id)); }

Rib *Scope::pop() {
  Rib *r = peek();
  stack.pop_back();
  return r;
}

void Scope::insert(const adt::CanonicalPath &path, basic::NodeId id,
                   Location loc, Rib::RibKind kind) {
  peek()->insertName(path, id, loc, true /*shadow*/, kind);
}

void Resolver::pushNewNameRib(Rib *r) { nameRibs[r->getNodeId()] = r; }
void Resolver::pushNewTypeRib(Rib *r) { typeRibs[r->getNodeId()] = r; }
void Resolver::pushNewLabelRib(Rib *r) { labelRibs[r->getNodeId()] = r; }
void Resolver::pushNewMaroRib(Rib *r) { macroRibs[r->getNodeId()] = r; }

Resolver::Resolver() noexcept
    : mappings(mappings::Mappings::get()), tyctx(TypeCheckContext::get()),
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
    mappings::Mappings::get()->insertModule(mod.get());
    pushNewModuleScope(modNodeId);
    resolveModule(mod, prefix, canonicalPrefix);
    popModuleScope();
    break;
  }
  case VisItemKind::ExternCrate: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::UseDeclaration: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::Function: {
    resolveFunction(std::static_pointer_cast<Function>(visItem), prefix,
                    canonicalPrefix);
    break;
  }
  case VisItemKind::TypeAlias: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::Struct: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::Enumeration: {
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::Union: {
    assert(false && "to be handled later");
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
    assert(false && "to be handled later");
    break;
  }
  case VisItemKind::Implementation: {
    resolveImplementation(
        std::static_pointer_cast<ast::Implementation>(visItem), prefix,
        canonicalPrefix);
    break;
  }
  case VisItemKind::ExternBlock: {
    assert(false && "to be handled later");
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
                                const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveInherentImpl(std::shared_ptr<ast::InherentImpl>,
                                   const adt::CanonicalPath &prefix,
                                   const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveTraitImpl(std::shared_ptr<ast::TraitImpl>,
                                const adt::CanonicalPath &prefix,
                                const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

void Resolver::resolveVisibility(std::optional<ast::Visibility> vis) {
  // FIXME
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver

// items: canoncialPath

// namespaces: types, macros, values, lifeftimes
