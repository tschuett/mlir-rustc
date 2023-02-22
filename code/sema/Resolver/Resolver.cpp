#include "Resolver.h"

#include "ADT/CanonicalPath.h"
#include "AST/Implementation.h"
#include "Basic/Ids.h"

#include <memory>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;

namespace rust_compiler::sema::resolver {

void Resolver::addModule(std::shared_ptr<ast::Module> mod, basic::NodeId nodeId,
                         const adt::CanonicalPath &path) {
  modules.insert_or_assign(nodeId, mod);
}

void Resolver::resolveCrate(std::shared_ptr<ast::Crate> crate) {
  for (auto &item : crate->getItems()) {
    switch (item->getItemKind()) {
    case ItemKind::VisItem: {
      resolveVisItem(std::static_pointer_cast<VisItem>(item));
      break;
    }
    case ItemKind::MacroItem: {
      resolveMacroItem(std::static_pointer_cast<MacroItem>(item));
      break;
    }
    }
  }
}

void Resolver::resolveVisItem(std::shared_ptr<ast::VisItem> visItem) {
  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    std::shared_ptr<ast::Module> mod =
        std::static_pointer_cast<Module>(visItem);
    basic::NodeId modNodeId = getNextNodeId();
    ScopedCanonicalPathScope scope = {&scopedPath, modNodeId,
                                      mod->getModuleName()};
    // setName(nodeId, scopedPath.getCurrentPath());
    addModule(mod, modNodeId, scopedPath.getCurrentPath());
    break;
  }
  case VisItemKind::ExternCrate: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::UseDeclaration: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::Function: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::TypeAlias: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::Struct: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::Enumeration: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::Union: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::ConstantItem: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::StaticItem: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::Trait: {
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::Implementation: {
    resolveImplementation(
        std::static_pointer_cast<ast::Implementation>(visItem));
    // setName(nodeId, canonicalPath);
    break;
  }
  case VisItemKind::ExternBlock: {
    // setName(nodeId, canonicalPath);
    break;
  }
  }
}

void Resolver::resolveImplementation(
    std::shared_ptr<ast::Implementation> impl) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    resolveInherentImpl(std::static_pointer_cast<ast::InherentImpl>(impl));
    break;
  }
  case ImplementationKind::TraitImpl: {
    resolveTraitImpl(std::static_pointer_cast<ast::TraitImpl>(impl));
    break;
  }
  }
}

void Resolver::resolveMacroItem(std::shared_ptr<ast::MacroItem>) {}

basic::NodeId Resolver::getNextNodeId() { return ++nodeId; }

void Resolver::resolveInherentImpl(std::shared_ptr<ast::InherentImpl>) {}

void Resolver::resolveTraitImpl(std::shared_ptr<ast::TraitImpl>) {}

} // namespace rust_compiler::sema::resolver

// items: canoncialPath

// namespaces: types, macros, values, lifeftimes
