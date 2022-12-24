#include "Toml/TokenStream.h"

#include <algorithm>

namespace rust_compiler::toml {

void TokenStream::append(Token tok) {
  tokens.push_back(tok);
}

std::span<Token> TokenStream::getViewAt(size_t offset) {
  return std::span<Token>(tokens).subspan(offset);
}

} // namespace rust_compiler::toml
