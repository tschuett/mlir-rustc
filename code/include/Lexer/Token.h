#pragma once

#include "Location.h"

#include <string>

// https://doc.rust-lang.org/reference/tokens.html

namespace rust_compiler::lexer {

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
  Plus,
  Keyword
};

class Token {
  rust_compiler::Location loc;
  TokenKind kind;
  std::string id;

public:
  Token(rust_compiler::Location loc, TokenKind tk) : loc(loc), kind(tk){};
  Token(rust_compiler::Location loc, TokenKind tk, std::string_view id)
      : loc(loc), kind(tk), id(id){};

  TokenKind getKind() const { return kind; }

  bool isUseToken() const;
  bool isPubToken() const;
  bool isCrateToken() const;
  bool isSelfToken() const;
  bool isSuperToken() const;
  bool isInToken() const;
  bool isDoubleColon() const;
  bool isIdentifier() const;
  bool isAs() const;

  std::string getIdentifier() const { return id; }

  rust_compiler::Location getLocation() const { return loc; }
};

std::string Token2String(TokenKind kind);

} // namespace rust_compiler::lexer
