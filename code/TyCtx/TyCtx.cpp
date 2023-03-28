#include "TyCtx/TyCtx.h"

#include "ADT/CanonicalPath.h"
#include "AST/AssociatedItem.h"
#include "AST/Crate.h"
#include "AST/EnumItem.h"
#include "AST/ExternalItem.h"
#include "Basic/Ids.h"

#include "../sema/TypeChecking/TypeChecking.h"

#include <memory>
#include <optional>

using namespace rust_compiler::basic;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::tyctx {

std::optional<std::string> TyCtx::getCrateName(CrateNum cnum) {
  auto it = astCrateMappings.find(cnum);
  if (it == astCrateMappings.end())
    return std::nullopt;

  return it->second->getCrateName();
}

TyCtx *TyCtx::get() {
  static std::unique_ptr<TyCtx> instance;
  if (!instance)
    instance = std::unique_ptr<TyCtx>(new TyCtx());

  return instance.get();
}

NodeId TyCtx::getNextNodeId() {
  auto it = nodeIdIter;
  ++nodeIdIter;
  return it;
}

void TyCtx::insertModule(ast::Module *mod) { assert(false); }

ast::Module *TyCtx::lookupModule(basic::NodeId id) {
  auto it = modules.find(id);
  if (it == modules.end())
    return nullptr;
  return it->second;
}

basic::CrateNum TyCtx::getCurrentCrate() const { return currentCrateNum; }

void TyCtx::setCurrentCrate(basic::CrateNum crateNum) {
  currentCrateNum = crateNum;
}

bool TyCtx::isModule(NodeId id) {
  return moduleChildItems.find(id) != moduleChildItems.end();
}

std::optional<adt::CanonicalPath>
TyCtx::lookupModuleChild(NodeId module, std::string_view itemName) {
  std::optional<std::vector<adt::CanonicalPath>> children =
      lookupModuleChildrenItems(module);
  if (!children)
    return std::nullopt;

  for (auto &child : *children) {
    std::string asString = child.asString();
    if (asString == itemName)
      return child;
  }
  return std::nullopt;
}

std::optional<std::vector<adt::CanonicalPath>>
TyCtx::lookupModuleChildrenItems(basic::NodeId module) {
  auto it = moduleChildItems.find(module);
  if (it == moduleChildItems.end())
    return std::nullopt;

  return it->second;
}

bool TyCtx::isCrate(NodeId nod) const {
  for (const auto &it : astCrateMappings) {
    NodeId crateNodeId = it.second->getNodeId();
    if (crateNodeId == nod)
      return true;
  }
  return false;
}

void TyCtx::insertASTCrate(ast::Crate *crate, CrateNum crateNum) {
  astCrateMappings.insert({crateNum, crate});
}

void TyCtx::insertBuiltin(NodeId id, NodeId ref, TyTy::BaseType *type) {
  nodeIdRefs[ref] = id;
  resolved[id] = type;
  builtins.push_back(std::unique_ptr<TyTy::BaseType>(type));
}

void TyCtx::insertType(const NodeIdentity &id, TyTy::BaseType *type) {
  resolved[id.getNodeId()] = type;
}

TyTy::BaseType *TyCtx::lookupBuiltin(std::string_view name) {
  for (auto &built : builtins) {
    if (built->toString() == name)
      return built.get();
  }
  return nullptr;
}

std::optional<TyTy::BaseType *> TyCtx::lookupType(basic::NodeId id) {
  auto it = resolved.find(id);
  if (it != resolved.end())
    return it->second;
  return std::nullopt;
}

std::optional<ast::Item *> TyCtx::lookupItem(basic::NodeId id) {
  auto it = itemMappings.find(id);
  if (it == itemMappings.end())
    return std::nullopt;

  return it->second;
}

std::optional<ast::ExternalItem *> TyCtx::lookupExternalItem(basic::NodeId id) {
  assert(false);
}

// void Resolver::insertImplementation(NodeId id, ast::Implementation *impl) {
//
// }
//
// std::optional<ast::Implementation *>
// TyCtx::lookupImplementation(basic::NodeId id) {
//   assert(false);
// }

std::optional<std::pair<ast::Enumeration *, ast::EnumItem *>>
TyCtx::lookupEnumItem(NodeId id) {
  auto it = enumItemsMappings.find(id);
  if (it == enumItemsMappings.end())
    return std::nullopt;

  return it->second;
}

void TyCtx::insertEnumItem(ast::Enumeration *parent, ast::EnumItem *item) {
  item->getNodeId();
  auto enumItem = lookupEnumItem(item->getNodeId());
  assert(not enumItem.has_value());
  NodeId id = item->getNodeId();
  enumItemsMappings[id] = {parent, item};
}

std::optional<ast::Implementation *>
TyCtx::lookupImplementation(basic::NodeId id) {
  auto it = implItemMapping.find(id);
  if (it == implItemMapping.end())
    return std::nullopt;

  return it->second;
}

std::optional<ast::AssociatedItem *>
TyCtx::lookupAssociatedItem(basic::NodeId implId) {
  assert(false);
}

} // namespace rust_compiler::tyctx
