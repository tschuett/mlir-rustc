#include "AST/Item.h"

#include "AST/Implementation.h"
#include "AST/Module.h"
#include "AST/VisItem.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

void CrateBuilder::emitItem(Item *item) {
  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    emitVisItem(static_cast<VisItem *>(item));
    break;
  }
  case ItemKind::MacroItem: {
    assert(false && "to be implemented");
  }
  }
}

void CrateBuilder::emitVisItem(VisItem *visItem) {

  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    emitModule(static_cast<ast::Module *>(visItem));
    break;
  }
  case VisItemKind::ExternCrate: {
    break;
  }
  case VisItemKind::UseDeclaration: {
    break;
  }
  case VisItemKind::Function: {
    emitFunction(static_cast<ast::Function *>(visItem));
    break;
  }
  case VisItemKind::TypeAlias: {
    break;
  }
  case VisItemKind::Struct: {
    emitStruct(static_cast<ast::Struct *>(visItem));
    break;
  }
  case VisItemKind::Enumeration: {
    break;
  }
  case VisItemKind::Union: {
    break;
  }
  case VisItemKind::ConstantItem: {
    break;
  }
  case VisItemKind::StaticItem: {
    break;
  }
  case VisItemKind::Trait: {
    break;
  }
  case VisItemKind::Implementation: {
    emitImplementation(static_cast<ast::Implementation *>(visItem));
    break;
  }
  case VisItemKind::ExternBlock: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
