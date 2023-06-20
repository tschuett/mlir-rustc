#include "TypeChecking.h"

#include "AST/MacroItem.h"
#include "Basic/Ids.h"
#include "Session/Session.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"

#include <cassert>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

// TypeCheckContext *TypeCheckContext::get() {
//   static TypeCheckContext *instance;
//   if (instance == nullptr)
//     instance = new TypeCheckContext();
//   return instance;
// }
//
// void TypeCheckContext::checkCrate(std::shared_ptr<ast::Crate>) {
//   assert(false && "to be done");
// }
//
// void TypeCheckContext::insertBuiltin(basic::NodeId nodeId,
//                                      basic::NodeId reference,
//                                      TyTy::BaseType *type) {
//   nodeToTypeReference[reference] = nodeId;
//   resolvedTypes[nodeId] = type;
//   builtinTypes.push_back(std::unique_ptr<TyTy::BaseType>(type));
// }

bool TypeResolver::queryInProgress(basic::NodeId id) {
  return queriesInProgress.find(id) != queriesInProgress.end();
}

void TypeResolver::insertQuery(basic::NodeId id) {
  queriesInProgress.insert(id);
}

void TypeResolver::queryCompleted(basic::NodeId id) {
  queriesInProgress.erase(id);
}

std::optional<TyTy::BaseType *> TypeResolver::queryType(basic::NodeId id) {
  if (queryInProgress(id))
    return std::nullopt;

  if (auto t = tcx->lookupType(id))
    return t;

  insertQuery(id);

  // enum item
  std::optional<std::pair<ast::Enumeration *, ast::EnumItem *>> enumItem =
      tcx->lookupEnumItem(id);
  if (enumItem) {
    Enumeration *enuM = enumItem->first;
    // EnumItem *item = enumItem->second;

    TyTy::BaseType *type = checkEnumerationPointer(enuM);

    queryCompleted(id);

    return type;
  }

  // plain item
  std::optional<Item *> item = tcx->lookupItem(id);
  if (item) {
    TyTy::BaseType *result = checkItemPointer(*item);
    queryCompleted(id);
    return result;
  }

  // associated item
  std::optional<Implementation *> impl = tcx->lookupImplementation(id);
  if (impl) {
    std::optional<std::pair<NodeId, AssociatedItem *>> asso =
        tcx->lookupAssociatedItem((*impl)->getNodeId());
    assert(asso.has_value());

    TyTy::BaseType *result = checkAssociatedItemPointer((*asso).second, *impl);
    queryCompleted(id);
    return result;
  }

  // implblock

  // extern item
  std::optional<ExternalItem *> external = tcx->lookupExternalItem(id);
  if (external) {
    TyTy::BaseType *result = checkExternalItemPointer(*external);
    queryCompleted(id);
    return result;
  }

  // more?
  queryCompleted(id);
  return std::nullopt;
}

TypeResolver::TypeResolver(resolver::Resolver *r) {
  tcx = rust_compiler::session::session->getTypeContext();
  resolver = r;
}

void TypeResolver::checkCrate(std::shared_ptr<ast::Crate> crate) {
  for (auto &item : crate->getItems()) {
    switch (item->getItemKind()) {
    case ItemKind::VisItem: {
      checkVisItem(std::static_pointer_cast<VisItem>(item));
      break;
    case ItemKind::MacroItem: {
      checkMacroItem(std::static_pointer_cast<ast::MacroItem>(item));
      break;
    }
    }
    }
  }

  // FIXME
}

TyTy::BaseType *TypeResolver::peekReturnType() {
  assert(!returnTypeStack.empty());
  return returnTypeStack.back().second;
}

void TypeResolver::pushReturnType(TypeCheckContextItem item,
                                  TyTy::BaseType *returnType) {
  assert(returnType != nullptr);
  returnTypeStack.push_back({std::move(item), returnType});
}

void TypeResolver::popReturnType() {
  assert(!returnTypeStack.empty());
  returnTypeStack.pop_back();
}

TypeCheckContextItem &TypeResolver::peekContext() {
  assert(!returnTypeStack.empty());
  return returnTypeStack.back().first;
}

TyTy::FunctionType *TypeCheckContextItem::getContextType() {
  NodeId id = basic::UNKNOWN_NODEID;
  switch (kind) {
  case ItemKind::Function:
    id = std::get<ast::Function *>(item)->getNodeId();
    break;
  }

  std::optional<TyTy::BaseType *> type =
      rust_compiler::session::session->getTypeContext()->lookupType(id);
  assert(type.has_value());
  return static_cast<TyTy::FunctionType *>(*type);
}

} // namespace rust_compiler::sema::type_checking
