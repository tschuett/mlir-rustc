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
  }
}

} // namespace rust_compiler::sema
