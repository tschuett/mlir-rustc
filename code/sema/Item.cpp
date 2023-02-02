#include "AST/Module.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::walkItem(std::shared_ptr<ast::Item> item) {

  walkOuterAttributes(item->getOuterAttributes());

  walkVisItem(item->getVisItem());
}

void Sema::analyzeItemDeclaration(std::shared_ptr<ast::Node> item) {
  assert(false);
}

} // namespace rust_compiler::sema
