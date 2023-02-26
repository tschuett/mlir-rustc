#pragma once

#include "AST/AST.h"
#include "Lexer/Token.h"

#include <optional>

namespace rust_compiler::ast {

class DelimTokenTree;

class TokenTree : public Node {
  std::optional<lexer::Token> token;
  std::optional<DelimTokenTree> tree;

public:
  TokenTree(Location loc) : Node(loc) {}

  void setToken(const lexer::Token &t) { token = t; }
  void setTree(const DelimTokenTree &delim) { tree = delim; }
};

} // namespace rust_compiler::ast
