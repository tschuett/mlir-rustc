#include "AST/OuterAttributes.h"

namespace rust_compiler::ast {

size_t OuterAttributes::getTokens() {
  size_t count = 0;

  for(auto& attr: outerAttributes)
    count += attr->getTokens();

  return count;
}

} // namespace rust_compiler::ast
