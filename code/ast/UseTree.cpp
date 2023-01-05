#include "AST/UseTree.h"

namespace rust_compiler::ast::use_tree {

void SimplePathDoubleColonStar::setPath(SimplePath path) {}

size_t DoubleColonStar::getTokens() {
  return 2; // :: *
}

size_t Star::getTokens() {
  return 1; // *
}

size_t PathList::getTokens() {
  size_t size = 0;
  for (auto &el : elements) {
    size += el->getTokens();
  }

  return 2 + 2 * size - 1; // { ... }
}

void PathList::addTree(std::shared_ptr<UseTree> tree) {
  elements.push_back(tree);
}

void SimplePathDoubleColonWithPathList::setPathList(PathList _path) {
  list = _path;
}

/// SimplePath :: { ... };
size_t SimplePathDoubleColonWithPathList::getTokens() {
  return 2 + list.getTokens() + 1;
}

void SimplePathNode::setSimplePath(SimplePath _path) { path = _path; }

size_t SimplePathNode::getTokens() { return path.getTokens(); }

} // namespace rust_compiler::ast::use_tree