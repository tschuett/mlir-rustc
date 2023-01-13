#include "AST/Module.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::walkItem(std::shared_ptr<ast::Item> item) {

  walkOuterAttributes(item->getOuterAttributes());

  walkVisItem(item->getVisItem());
}

} // namespace rust_compiler::sema
