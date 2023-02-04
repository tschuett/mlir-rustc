#include "AST/Module.h"
#include "AST/VisItem.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

void CrateBuilder::emitItem(std::shared_ptr<Item> &item) {

  emitVisItem(item->getVisItem());
}

void CrateBuilder::emitVisItem(std::shared_ptr<VisItem> visItem) {

  switch (visItem->getKind()) {
  case VisItemKind::Module: {
    emitModule(std::static_pointer_cast<ast::Module>(visItem));
    break;
  }
  case VisItemKind::ExternCrate: {
    break;
  }
  case VisItemKind::UseDeclaration: {
    break;
  }
  case VisItemKind::Function: {
    emitFunction(std::static_pointer_cast<ast::Function>(visItem));
    break;
  }
  case VisItemKind::TypeAlias: {
    break;
  }
  case VisItemKind::Struct: {
    // FIXME
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
    // FIXME
    break;
  }
  case VisItemKind::ExternBlock: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
