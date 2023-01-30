#include "AST/Item.h"
#include "AST/OuterAttributes.h"
#include "AST/VisItem.h"

#include <cassert>

namespace rust_compiler::ast {

void Item::setOuterAttributes(std::shared_ptr<OuterAttributes> outer) {
  outerAttributes = outer;
}

void Item::setVisItem(std::shared_ptr<VisItem> _visItem) { visItem = _visItem; }

size_t Item::getTokens() {
  size_t count = 0;

  if (outerAttributes)
    count += outerAttributes->getTokens();

  return count + visItem->getTokens();
}

std::shared_ptr<VisItem> Item::getVisItem() const { return visItem; }

std::shared_ptr<OuterAttributes> Item::getOuterAttributes() const {
  return outerAttributes;
}

} // namespace rust_compiler::ast
