#pragma once

#include "Lexer/KeyWords.h"
#include "Location.h"

#include <string>

// https://doc.rust-lang.org/reference/tokens.html

namespace rust_compiler::lexer {

enum class IntegerKind {
  I8,
  I16,
  I32,
  I64,
  I128,
  ISize,
  U8,
  U16,
  U32,
  U64,
  U128,
  USize
};

enum class FloatKind { F32, F64 };

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
  Keyword,
  Not,
  Integer,
  Float
};

class Token {
  rust_compiler::Location loc;
  TokenKind kind;
  std::string id;
  IntegerKind ik;
  FloatKind fk;
  KeyWordKind kw;

public:
  Token(rust_compiler::Location loc, TokenKind tk) : loc(loc), kind(tk){};
  Token(rust_compiler::Location loc, TokenKind tk, std::string_view id)
      : loc(loc), kind(tk), id(id){};

  Token(rust_compiler::Location loc, KeyWordKind kw, std::string_view id)
      : loc(loc), kind(TokenKind::Keyword), id(id), kw(kw){};

  Token(rust_compiler::Location loc, IntegerKind ik)
      : loc(loc), kind(TokenKind::Integer), ik(ik){};

  Token(rust_compiler::Location loc, FloatKind fk)
      : loc(loc), kind(TokenKind::Float), fk(fk){};

  TokenKind getKind() const { return kind; }
  IntegerKind getIntegerKind() const { return ik; }
  FloatKind getFloatKind() const { return fk; }
  KeyWordKind getKeyWordKind() const { return kw; }

  bool isKeyWord() const;

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
