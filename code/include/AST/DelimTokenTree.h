#pragma once

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class TokenTree;

class DelimTokenTree {
  // mode
  std::vector<std::shared_ptr<TokenTree>> trees;
};

} // namespace rust_compiler::ast
