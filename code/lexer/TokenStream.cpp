#include "TokenStream.h"

namespace rust_compiler {

void TokenStream::append(Token tk) { tokens.push_back(tk); }

std::span<Token> TokenStream::getAsView() {
  return std::span<Token>(tokens);
}

} // namespace rust_compiler
