#pragma once

#include "Lexer/TokenStream.h"

#include <string_view>

namespace rust_compiler::lexer {

TokenStream lex(std::string_view code, std::string_view fileName);

class Lexer {
  std::string chars;
  uint32_t remaining;
  TokenStream tokenStream;
public:
  void lex();


private:
  std::optional<char> bump();

  Token advanceToken();
};

} // namespace rust_compiler::lexer
