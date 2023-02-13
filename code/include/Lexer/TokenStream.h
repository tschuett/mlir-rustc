#pragma once

#include "Lexer/Token.h"

#include <span>
#include <vector>

namespace rust_compiler::lexer {

class TokenStream {
  std::vector<Token> tokens;

public:
  void append(Token tk);
  std::span<Token> getAsView();

  void print(unsigned limit);

  size_t getLength() const { return tokens.size(); }

  Token getAt(size_t);
};

} // namespace rust_compiler::lexer
