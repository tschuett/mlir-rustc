#include "AST/Item.h"

#include <cassert>

namespace rust_compiler::ast {

void Item::setOuterAttributes(std::shared_ptr<OuterAttributes> outer) {
  outerAttributes = outer;
}

void Item::setVisItem(std::shared_ptr<VisItem> _visItem) { visItem = _visItem; }

} // namespace rust_compiler::ast
