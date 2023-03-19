#include "TypeChecking.h"

#include "AST/MacroItem.h"
#include "TyCtx/TyCtx.h"

#include <cassert>

using namespace rust_compiler::ast;

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

TypeResolver::TypeResolver() { tcx = tyctx::TyCtx::get(); }

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

} // namespace rust_compiler::sema::type_checking
