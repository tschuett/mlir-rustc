#include "AST/Item.h"

#include "AST/OuterAttribute.h"
#include "AST/VisItem.h"

#include <cassert>

namespace rust_compiler::ast {

void Item::setOuterAttributes(std::span<OuterAttribute> outer) {
  outerAttributes = {outer.begin(), outer.end()};
}

std::span<OuterAttribute> Item::getOuterAttributes()  {
  return outerAttributes;
}

} // namespace rust_compiler::ast
