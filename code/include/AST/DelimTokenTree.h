#pragma once

#include "AST/AST.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class TokenTree;

class DelimTokenTree : public Node {
  // mode
  std::vector<std::shared_ptr<TokenTree>> trees;

public:
  DelimTokenTree(Location loc) : Node(loc){};
};

} // namespace rust_compiler::ast
