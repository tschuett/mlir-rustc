#pragma once

#include "Toml/Token.h"

#include <vector>

namespace rust_compiler::toml {

class TokenStream {
  std::vector<Token> tokens;

public:
  void append(Token tok);
};

} // namespace rust_compiler::toml
