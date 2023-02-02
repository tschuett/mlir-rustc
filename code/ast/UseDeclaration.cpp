#include "AST/UseDeclaration.h"

#include "AST/UseTree.h"

namespace rust_compiler::ast {

size_t UseDeclaration::getTokens() {
  return 2 + tree->getTokens(); //  +  FIXME   use <tree> ;
}

void UseDeclaration::setComponent(std::shared_ptr<use_tree::UseTree> _tree) {
  tree = _tree;
}

} // namespace rust_compiler::ast
