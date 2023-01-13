#include "AST/Item.h"

#include <cassert>

namespace rust_compiler::ast {

void Item::setOuterAttributes(std::shared_ptr<OuterAttributes> outer) {
  outerAttributes = outer;
}

void Item::setVisItem(std::shared_ptr<VisItem> _visItem) { visItem = _visItem; }

size_t Item::getTokens() {
  assert(false);

  return 1;
}

} // namespace rust_compiler::ast
