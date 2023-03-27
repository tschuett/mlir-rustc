#include "TyCtx/TyCtx.h"

#include "ADT/CanonicalPath.h"
#include "AST/Crate.h"
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

std::optional<TyTy::BaseType *> TyCtx::queryType(basic::NodeId id,
                                                 TypeResolver *typeResolver) {
  if (queryInProgress(id))
    return std::nullopt;

  if (auto t = lookupType(id))
    return t;

  insertQuery(id);

  // enum item
  std::optional<std::pair<ast::Enumeration *, ast::EnumItem *>> enumItem =
      lookupEnumItem(id);
  if (enumItem) {
    Enumeration *enuM = enumItem->first;
    // EnumItem *item = enumItem->second;

    TyTy::BaseType *type = typeResolver->checkEnumerationPointer(enuM);

    queryCompleted(id);

    return type;
  }

  // plain item
  std::optional<Item *> item = lookupItem(id);
  if (item) {
    TyTy::BaseType *result = typeResolver->checkItemPointer(*item);
    queryCompleted(id);
    return result;
  }

  // implementation
  std::optional<Implementation *> impl = lookupImplementation(id);
  if (impl) {
    TyTy::BaseType *result = typeResolver->checkImplementationPointer(*impl);
    queryCompleted(id);
    return result;
  }

  // extern item
  std::optional<ExternalItem *> external = lookupExternalItem(id);
  if (external) {
    TyTy::BaseType *result = typeResolver->checkExternalItemPointer(*external);
    queryCompleted(id);
    return result;
  }

  // more?
  queryCompleted(id);
  return std::nullopt;
}

bool TyCtx::queryInProgress(basic::NodeId id) {
  return queriesInProgress.find(id) != queriesInProgress.end();
}

void TyCtx::insertQuery(basic::NodeId id) { queriesInProgress.insert(id); }

void TyCtx::queryCompleted(basic::NodeId id) { queriesInProgress.erase(id); }

std::optional<ast::Item *> TyCtx::lookupItem(basic::NodeId id) {
  assert(false);
}
std::optional<ast::ExternalItem *> TyCtx::lookupExternalItem(basic::NodeId id) {
  assert(false);
}

std::optional<ast::Implementation *>
TyCtx::lookupImplementation(basic::NodeId id) {
  assert(false);
}

std::optional<std::pair<ast::Enumeration *, ast::EnumItem *>>
TyCtx::lookupEnumItem(NodeId id) {
  assert(false);
}

} // namespace rust_compiler::tyctx
