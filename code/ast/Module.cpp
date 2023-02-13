#include "AST/Module.h"

namespace rust_compiler::ast {

// std::span<std::shared_ptr<Function>> Module::getFuncs() {
//   return std::span<std::shared_ptr<Function>>(funs);
// }

std::span<std::shared_ptr<Item>> Module::getItems() { return items; }

void Module::addItem(std::shared_ptr<Item> item) { items.push_back(item); }

} // namespace rust_compiler::ast
