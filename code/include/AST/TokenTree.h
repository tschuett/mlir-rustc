#pragma once

#include "AST/AST.h"
#include "AST/DelimTokenTree.h"
#include "Lexer/Token.h"

#include <variant>

namespace rust_compiler::ast {

class DelimTokenTree;

class TokenTree : public Node {
  std::variant<lexer::Token, DelimTokenTree> contents;

public:
  TokenTree(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
