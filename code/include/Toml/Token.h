#pragma once

#include <string>

#include <optional>

namespace rust_compiler::toml {

enum class TokenKind {
  Hash,
  SquareOpen,
  Identifier,
  SquareClose,
  Integer,
  Equal,
  String,
  BraceOpen,
  Comma,
  BraceClose
};

class Token {
  TokenKind kind;
  std::optional<std::string> id;

public:
  Token(TokenKind kind) : kind(kind) {}
  Token(TokenKind kind, std::string_view id) : kind(kind), id(id) {}

  TokenKind getKind() const { return kind; }
  std::string getString() const { return *id; } // may crash
};

} // namespace rust_compiler::toml
