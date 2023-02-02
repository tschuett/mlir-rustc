#pragma once

//#include "AST/DelimTokenTree.h"
#include "Lexer/Token.h"

namespace rust_compiler::ast {

class DelimTokenTree;

class TokenTree {
  std::variant<lexer::Token, DelimTokenTree> contents;
};

} // namespace rust_compiler::ast
