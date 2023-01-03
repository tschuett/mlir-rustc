#include "AST/Module.h"

namespace rust_compiler::ast {

size_t Module::getTokens() {
  return 3; //  +  FIXME
}

std::span<std::shared_ptr<Function>> Module::getFuncs() {
  return std::span<std::shared_ptr<Function>>(funs);
}

void Module::addItem(std::shared_ptr<Item> item) { items.push_back(item); }

void Module::setVisibility(Visibility vis) { assert(false); }

} // namespace rust_compiler::ast
