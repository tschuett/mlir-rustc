#pragma once

#include "Location.h"

#include <string>

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
  Plus
};

class Token {
  LocationAttr loc;
  TokenKind kind;
  std::string id;

public:
  Token(LocationAttr loc, TokenKind tk) : loc(loc), kind(tk){};
  Token(LocationAttr loc, TokenKind tk, std::string_view id)
      : loc(loc), kind(tk), id(id){};

  TokenKind getKind() const { return kind; }

  bool isUseToken() const;
  bool isPubToken() const;
  bool isCrateToken() const;
  bool isSelfToken() const;
  bool isSuperToken() const;
  bool isInToken() const;

  std::string getIdentifier() const { return id; }
};

} // namespace rust_compiler::lexer
