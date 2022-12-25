#pragma once

#include "Token.h"

#include <span>
#include <vector>

namespace rust_compiler {

class TokenStream {
  std::vector<Token> tokens;

public:
  void append(Token tk);
  std::span<Token> getAsView();
};

} // namespace rust_compiler
