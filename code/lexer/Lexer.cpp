#include "Lexer.h"

#include "TokenStream.h"

namespace rust_compiler {

void TokenStream::append(Token tk) { tokens.push_back(tk); }

TokenStream lex(std::string_view code) {
  TokenStream ts;

  while (code.size() > 0) {
   if (code.starts_with("!")) {
      ts.append(Token(TokenKind::Exclaim));
      code.remove_prefix(1);
  } else if (code.starts_with("::")) {
      ts.append(Token(TokenKind::DoubleColon));
      code.remove_prefix(2);
    } else if (code.starts_with("#")) {
      ts.append(Token(TokenKind::Hash));
      code.remove_prefix(1);
    } else if (code.starts_with("[")) {
      ts.append(Token(TokenKind::SquareOpen));
      code.remove_prefix(1);
    } else if (code.starts_with(",")) {
      ts.append(Token(TokenKind::Comma));
      code.remove_prefix(1);
    } else {
      printf("unknown token: %s\n", code.data());
      exit(EXIT_FAILURE);
    }
  }

  return ts;
}

} // namespace rust_compiler


// https://github.com/thepowersgang/mrustc/blob/master/src/parse/lex.cpp
