#pragma once

#include "AST/AST.h"
#include "AST/TokenTree.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class DelimTokenTree : public Node {
  // mode
  std::vector<TokenTree> trees;

public:
  DelimTokenTree(Location loc) : Node(loc){};

  void addTree(const TokenTree &t) { trees.push_back(t); }
};

} // namespace rust_compiler::ast
