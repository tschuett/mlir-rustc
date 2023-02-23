#pragma once

#include "Lexer/KeyWords.h"
#include "Location.h"

#include <string>
#include <variant>

// https://doc.rust-lang.org/reference/tokens.html
// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/token/enum.TokenKind.html

namespace rust_compiler::lexer {

using uint128_t = unsigned __int128;
using int128_t = __int128;

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
  RArrow,
  // LessThan,
  //  DoubleGreaterThan,
  //  GreaterThan,
  Eq,
  Dot,
  QMark,
  Hash,
  // Exclaim,
  SquareOpen,
  SquareClose,
  Comma,
  Identifier,
  ParenOpen,
  ParenClose,
  Semi,
  BraceOpen,
  BraceClose,
  // Amp,
  // DoubleAmp,
  String,
  // Pipe,
  Star,
  // Dash,
  Char,
  Plus,
  Keyword,
  Not,
  //  Integer,
  //  Float,
  Minus,
  Slash,
  Percent,
  And,
  AndAnd,
  Or,
  OrOr,
  PathSep,
  Caret,
  Shl,
  Shr,
  DotDot,
  DotDotDot,
  DotDotEq,
  Lt,
  EqEq,
  Ne,
  Gt,
  Ge,
  Le,
  Underscore,
  LIFETIME_TOKEN,
  LIFETIME_OR_LABEL,
  // DollarUnderScore,
  CHAR_LITERAL,
  QUOTE_ESCAPE,
  ASCII_ESCAPE,
  UNICODE_ESCAPE,
  STRING_LITERAL,
  RAW_STRING_LITERAL,
  BYTE_LITERAL,
  BYTE_STRING_LITERAL,
  RAW_BYTE_STRING_LITERAL,
  INTEGER_LITERAL,
  FLOAT_LITERAL,
  RESERVED_NUMBER,
  Eof,
  At,
  StarEq,
  PlusEq,
  MinusEq,
  SlashEq,
  PercentEq,
  AndEq,
  ShlEq,
  CaretEq,
  ShrEq,
  OrEq,
  FatArrow,
};

class Token {
  rust_compiler::Location loc;
  TokenKind kind;
  std::string storage;
  //  IntegerKind ik;
  //  FloatKind fk;
  KeyWordKind kw;

  //  std::variant<int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
  //  int64_t,
  //               uint64_t, int128_t, uint128_t, float, double, std::string>
  //      content;

public:
  Token(rust_compiler::Location loc, TokenKind tk) : loc(loc), kind(tk){};
  Token(rust_compiler::Location loc, TokenKind tk, std::string_view id)
      : loc(loc), kind(tk), storage(id){};

  Token(rust_compiler::Location loc, KeyWordKind kw, std::string_view id)
      : loc(loc), kind(TokenKind::Keyword), storage(id), kw(kw){};

  //  Token(rust_compiler::Location loc, IntegerKind ik)
  //      : loc(loc), kind(TokenKind::Integer), ik(ik){};
  //
  //  Token(rust_compiler::Location loc, FloatKind fk)
  //      : loc(loc), kind(TokenKind::Float), fk(fk){};

  TokenKind getKind() const { return kind; }
  //  IntegerKind getIntegerKind() const { return ik; }
  // FloatKind getFloatKind() const { return fk; }
  KeyWordKind getKeyWordKind() const { return kw; }

  bool isKeyWord() const { return kind == TokenKind::Keyword; }

  bool isUseToken() const;
  bool isPubToken() const;
  bool isCrateToken() const;
  bool isSelfToken() const;
  bool isSuperToken() const;
  bool isInToken() const;
  bool isDoubleColon() const;
  bool isIdentifier() const;
  bool isAs() const;

  std::string getIdentifier() const { return storage; }

  rust_compiler::Location getLocation() const { return loc; }

  std::string getLiteral() const { return storage; }
};

std::string Token2String(TokenKind kind);

} // namespace rust_compiler::lexer
