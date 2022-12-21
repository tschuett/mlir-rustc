#pragma once

#include "Token.h"

#include <vector>
namespace rust_compiler {

class TokenStream {
  std::vector<Token> tokens;

public:
  void append(Token tk);
};
} // namespace rust_compiler
