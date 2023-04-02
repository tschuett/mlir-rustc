#include "Lexer/TokenStream.h"

namespace rust_compiler::lexer {

void TokenStream::append(Token tk) { tokens.push_back(tk); }

std::span<Token> TokenStream::getAsView() { return std::span<Token>(tokens); }

void TokenStream::print(unsigned limit) {
  unsigned idx = 0;

  while (idx < limit and idx < tokens.size()) {
    if (tokens[idx].isIdentifier()) {
      printf("ident: x%sx ", tokens[idx].getIdentifier().toString().c_str());
    } else {
      printf("token: %s ", Token2String(tokens[idx].getKind()).c_str());
    }
    ++idx;
  }

  printf("\n");
}

} // namespace rust_compiler::lexer
