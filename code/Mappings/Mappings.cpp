#include "Mappings/Mappings.h"

#include "ADT/CanonicalPath.h"
#include "Basic/Ids.h"
#include "AST/Crate.h"

#include <memory>
#include <optional>

using namespace rust_compiler::basic;
using namespace rust_compiler::ast;

namespace rust_compiler::mappings {

Mappings *Mappings::get() {
  static std::unique_ptr<Mappings> instance;
  if (!instance)
    instance = std::unique_ptr<Mappings>(new Mappings());

  return instance.get();
}

NodeId Mappings::getNextNodeId() {
  auto it = nodeIdIter;
  ++nodeIdIter;
  return it;
}

void Mappings::insertModule(ast::Module *mod) { assert(false); }

ast::Module *Mappings::lookupModule(basic::NodeId id) {
  auto it = modules.find(id);
  if (it == modules.end())
    return nullptr;
  return it->second;
}

basic::CrateNum Mappings::getCurrentCrate() const { return currentCrateNum; }

void Mappings::setCurrentCrate(basic::CrateNum crateNum) {
  currentCrateNum = crateNum;
}

bool Mappings::isModule(NodeId id) {
  return moduleChildItems.find(id) != moduleChildItems.end();
}

std::optional<adt::CanonicalPath>
Mappings::lookupModuleChild(NodeId module, std::string_view itemName) {
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
Mappings::lookupModuleChildrenItems(basic::NodeId module) {
  auto it = moduleChildItems.find(module);
  if (it == moduleChildItems.end())
    return std::nullopt;

  return it->second;
}

bool Mappings::isCrate(NodeId nod) const {
  for (const auto &it : astCrateMappings) {
    NodeId crateNodeId = it.second->getNodeId();
    if (crateNodeId == nod)
      return true;
  }
  return false;
}

void Mappings::insertASTCrate(ast::Crate *crate, CrateNum crateNum) {
  astCrateMappings.insert({crateNum, crate});
}

} // namespace rust_compiler::mappings
