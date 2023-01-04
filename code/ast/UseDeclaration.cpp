#include "AST/UseDeclaration.h"

namespace rust_compiler::ast {

size_t UseDeclaration::getTokens() {
  return 3; //  +  FIXME
}

void UseDeclaration::setComponent(std::shared_ptr<use_tree::UseTree> _tree) {
  tree = _tree;
}

} // namespace rust_compiler::ast
