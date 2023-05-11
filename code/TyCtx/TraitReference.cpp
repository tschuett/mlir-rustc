#include "TyCtx/TraitReference.h"

#include "AST/Item.h"
#include "AST/VisItem.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TyTy.h"

#include <memory>

using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::tyctx {

// void TraitReference::onResolved() {
//   for (auto &item : items)
//     item.onResolved();
// }
//
// void TraitItemReference::onResolved() {
//   switch (kind) {
//   case TraitItemKind::Constant: {
//     assert(false);
//     break;
//   }
//   case TraitItemKind::Function: {
//     auto fun = item->getFunction();
//     assert(fun->getItemKind() == ast::ItemKind::VisItem);
//     resolveFunction(
//         std::static_pointer_cast<ast::Function>(
//             std::static_pointer_cast<ast::VisItem>(item->getFunction()))
//             .get());
//     break;
//   }
//   case TraitItemKind::TypeAlias: {
//     assert(false);
//     break;
//   }
//   case TraitItemKind::Error: {
//     break;
//   }
//   }
// }
//
// void TraitItemReference::resolveFunction(ast::Function *) {
//   if (!isOptional)
//     return;
//
//   TyTy::BaseType *itemType = getType();
//   if (itemType->getKind() == TypeKind::Error)
//     return;
//
//   assert(itemType->getKind() == TypeKind::Function);
//
//   //TyTy::FunctionType *fn = static_cast<TyTy::FunctionType*>(itemType);
//   //TyTy::BaseType *retType = fn->getReturnType();
//   //tcx->pushReturnType(TypeCheckContextItem(), ret);
//
//   // FIXME
// }

std::string TraitReference::toString() const {
  if (isError())
    return "<trait-ref-error-node>";

  std::string itemBuf;
  for (auto &item : items) {
    itemBuf += item.toString() + ", ";
  }
  return "Trait: " + getIdentifier().toString() + "->" +
    /*trait->get_mappings().as_string() +*/ " [" + itemBuf + "]";
}

std::string TraitItemReference::toString() const {
  return "(" + traitItemTypeToString() + " " + identifier.toString() + " " + ")";
}

std::string TraitItemReference::traitItemTypeToString() const {
  switch (kind) {
  case TraitItemKind::Constant:
    return "CONST";
  case TraitItemKind::Function:
    return "FN";
  case TraitItemKind::TypeAlias:
    return "TYPE";
  case TraitItemKind::Error:
    return "ERROR";
  }
}

} // namespace rust_compiler::tyctx
