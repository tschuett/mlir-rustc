#include "TyCtx/TyCtx.h"

#include "ADT/CanonicalPath.h"
#include "AST/Crate.h"
#include "Basic/Ids.h"

#include <memory>
#include <optional>

using namespace rust_compiler::basic;
using namespace rust_compiler::ast;

namespace rust_compiler::tyctx {

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

} // namespace rust_compiler::tyctx
