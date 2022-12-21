#pragma once

#include "Toml/Token.h"

#include <vector>

#include <span>

namespace rust_compiler::toml {

class TokenStream {
  std::vector<Token> tokens;

public:
  void append(Token tok);

  std::span<Token> getViewAt(size_t offset);
};

} // namespace rust_compiler::toml
