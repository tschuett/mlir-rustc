#pragma once

#include "ADT/CanonicalPath.h"
#include "Basic/Ids.h"

#include <map>
#include <optional>

namespace rust_compiler::ast {
class Module;
class Item;
} // namespace rust_compiler::ast

namespace rust_compiler::mappings {

class Mappings {
public:
  static Mappings *get();

  basic::NodeId getNextNodeId();

  void insertModule(ast::Module *);

  ast::Module *lookupModule(basic::NodeId);

  void insertCanonicalPath(basic::NodeId id, const adt::CanonicalPath &path) {
    if (auto canPath = lookupCanonicalPath(id)) {
      if (canPath->isEqual(path))
        return;
      assert(canPath->getSize() >= path.getSize());
    }
    paths.emplace(id, path);
  }

  void insertChildItemToParentModuleMapping(basic::NodeId child,
                                            basic::NodeId parentModule) {
    childToParentModuleMap.insert({child, parentModule});
  }

  std::optional<adt::CanonicalPath> lookupCanonicalPath(basic::NodeId id) {
    auto it = paths.find(id);
    if (it == paths.end())
      return std::nullopt;
    return it->second;
  }

  void insertModuleChildItem(basic::NodeId module, adt::CanonicalPath child) {
    auto it = moduleChildItems.find(module);
    if (it == moduleChildItems.end())
      moduleChildItems.insert({module, {child}});
    else
      it->second.emplace_back(child);
  }

  basic::CrateNum getCurrentCrate() const;
  void setCurrentCrate(basic::CrateNum);

private:
  basic::CrateNum crateNumIter = 7;
  basic::NodeId nodeIdIter = 7;
  basic::CrateNum currentCrateNum = basic::UNKNOWN_CREATENUM;

  std::map<basic::NodeId, ast::Module *> modules;
  std::map<basic::NodeId, ast::Item *> items;
  std::map<basic::NodeId, adt::CanonicalPath> paths;

  // Maps each module's node id to a list of its children
  std::map<basic::NodeId, std::vector<basic::NodeId>> moduleChildMap;
  std::map<basic::NodeId, std::vector<adt::CanonicalPath>> moduleChildItems;
  std::map<basic::NodeId, basic::NodeId> childToParentModuleMap;
};

} // namespace rust_compiler::sema
