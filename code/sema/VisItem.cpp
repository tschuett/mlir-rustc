#include "AST/Module.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::walkVisItem(std::shared_ptr<ast::VisItem> item) {
  switch (item->getKind()) {
  case VisItemKind::Module: {
    break;
  }
  case VisItemKind::Function: {
    break;
  }
  case VisItemKind::UseDeclaration: {
    break;
  }
  case VisItemKind::ExternCrate: {
    break;
  }
  case VisItemKind::TypeAlias: {
    break;
  }
  case VisItemKind::Struct: {
    break;
  }
  case VisItemKind::ConstantItem: {
    break;
  }
  case VisItemKind::Enumeration: {
    break;
  }
  case VisItemKind::Union: {
    break;
  }
  case VisItemKind::StaticItem: {
    break;
  }
  case VisItemKind::Trait: {
    break;
  }
  case VisItemKind::Implementation: {
    break;
  }
  case VisItemKind::ExternBlock: {
    break;
  }
  }
}

} // namespace rust_compiler::sema
