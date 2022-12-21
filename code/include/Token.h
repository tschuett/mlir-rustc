#pragma once

#include <string>

namespace rust_compiler {

enum class TokenKind { DoubleColon, Hash, Exclaim, SquareOpen, Comma };

class Token {
  std::string s;
  TokenKind kind;

public:
  Token(TokenKind tk) : kind(tk){};

  TokenKind getKind() const { return kind; }
};

} // namespace rust_compiler
