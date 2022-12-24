#pragma once

#include "Toml/Token.h"

#include <span>
#include <vector>

namespace rust_compiler::toml {

class TokenStream {
  std::vector<Token> tokens;

public:
  void append(Token tok);

  std::span<Token> getViewAt(size_t offset);

  size_t getSize() const { return tokens.size(); }
};

} // namespace rust_compiler::toml
