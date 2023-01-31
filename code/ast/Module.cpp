#include "AST/Module.h"

namespace rust_compiler::ast {

size_t Module::getTokens() {
  switch (kind) {
  case ModuleKind::Module: {
    return 3 + vis.getTokens(); // mod <name> ;
  }
  case ModuleKind::ModuleTree: {
    size_t size = 0;
    for (auto &i : items) {
      size += i->getTokens();
    }
    // FIXME
    return size + 5 + vis.getTokens();
  }
  }
}

// std::span<std::shared_ptr<Function>> Module::getFuncs() {
//   return std::span<std::shared_ptr<Function>>(funs);
// }

std::span<std::shared_ptr<Item>> Module::getItems() { return items; }

void Module::addItem(std::shared_ptr<Item> item) { items.push_back(item); }

void Module::setVisibility(Visibility _vis) { vis = _vis; }

} // namespace rust_compiler::ast
