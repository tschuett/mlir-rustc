#include "AST/Module.h"
#include "Sema/Sema.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::walkItem(std::shared_ptr<ast::Item> item) {

  walkOuterAttributes(item->getOuterAttributes());

  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    walkVisItem(std::static_pointer_cast<VisItem>(item));
    break;
  }
  case ItemKind::MacroItem: {
    break;
  }
  }
}

void Sema::analyzeItemDeclaration(std::shared_ptr<ast::Node> item) {
  assert(false);
}

} // namespace rust_compiler::sema
