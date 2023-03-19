#pragma once

#include "ADT/CanonicalPath.h"
#include "Basic/Ids.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TyCtx.h"

#include "../../code/sema/TypeChecking/TyTy.h"

#include <map>
#include <optional>
#include <vector>

namespace rust_compiler::ast {
class Module;
class Item;
class Crate;
} // namespace rust_compiler::ast

namespace rust_compiler::tyctx {

using namespace rust_compiler::sema::type_checking;

class TyCtx {
public:
  static TyCtx *get();

  basic::NodeId getNextNodeId();

  void insertModule(ast::Module *);

  ast::Module *lookupModule(basic::NodeId);

  std::optional<adt::CanonicalPath>
  lookupModuleChild(basic::NodeId module, std::string_view item_name);

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

  bool isModule(basic::NodeId query);
  bool isCrate(basic::NodeId query) const;
  // void insertNodeToNode(basic::NodeId id, basic::NodeId ref);

  std::optional<std::vector<adt::CanonicalPath>>
  lookupModuleChildrenItems(basic::NodeId module);

  void insertASTCrate(ast::Crate *crate, basic::CrateNum crateNum);

  void insertBuiltin(basic::NodeId id, basic::NodeId ref, TyTy::BaseType *type);
  TyTy::BaseType *lookupBuiltin(std::string_view name);

  void insertType(const NodeIdentity& id, TyTy::BaseType *type);

  TyTy::BaseType *peekReturnType();
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

  std::map<basic::CrateNum, ast::Crate *> astCrateMappings;

  std::map<basic::NodeId, basic::NodeId> nodeIdRefs;
  std::map<basic::NodeId, TyTy::BaseType *> resolved;
  std::vector<std::unique_ptr<TyTy::BaseType>> builtins;
};

} // namespace rust_compiler::tyctx
