#pragma once

#include <string>

namespace rust_compiler {

enum class TokenKind {
  DoubleColon,
  Colon,
  ThinArrow,
  LessThan,
  DoubleGreaterThan,
  GreaterThan,
  Equals,
  Dot,
  QMark,
  Hash,
  Exclaim,
  SquareOpen,
  SquareClose,
  Comma,
  Identifier,
  ParenOpen,
  ParenClose,
  SemiColon,
  BraceOpen,
  BraceClose,
  Amp,
  DoubleAmp,
  String,
  Pipe,
  Star,
  Dash,
  Char,
  Plus
};

class Token {
  TokenKind kind;
  std::string s;

public:
  Token(TokenKind tk) : kind(tk){};
  Token(TokenKind tk, std::string_view id) : kind(tk), s(id){};

  TokenKind getKind() const { return kind; }

  bool isUseToken() const;
  bool isPubToken() const;
  bool isCrateToken() const;
  bool isSelfToken() const;
  bool isSuperToken() const;
  bool isInToken() const;

  std::string getIdentifier() const { return s; }
};

} // namespace rust_compiler
