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

  Token getAt(size_t at) const {
    assert(at < tokens.size());
    return tokens[at];
  }
};

} // namespace rust_compiler::lexer
