#include "AST/UseTree.h"

namespace rust_compiler::ast::use_tree {

void SimplePathDoubleColonStar::setPath(SimplePath path) {}


void PathList::addTree(std::shared_ptr<UseTree> tree) {
  elements.push_back(tree);
}

void SimplePathDoubleColonWithPathList::setPathList(PathList _path) {
  list = _path;
}


void SimplePathNode::setSimplePath(SimplePath _path) { path = _path; }


} // namespace rust_compiler::ast::use_tree
