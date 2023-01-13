#include "AST/Expression.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler {

void ModuleBuilder::emitItem(std::shared_ptr<ast::Item> item) {
  llvm::outs() << "emitItem"
               << "\n";
  emitVisItem(item->getVisItem());
}

} // namespace rust_compiler
