#include "Toml/Token.h"

namespace rust_compiler::toml {

std::string Token::toString() {

  std::string kindString;

  switch (kind) {
  case TokenKind::Hash: {
    kindString = "#";
    break;
  }
  case TokenKind::SquareOpen: {
    kindString = "[";
    break;
  }
  case TokenKind::Integer: {
    kindString = id;
    break;
  }
  case TokenKind::Identifier: {
    kindString = id;
    break;
  }
  case TokenKind::String: {
    kindString = id;
    break;
  }
  case TokenKind::BraceClose: {
    kindString = "}";
    break;
  }
  case TokenKind::BraceOpen: {
    kindString = "{";
    break;
  }
  case TokenKind::Comma: {
    kindString = ",";
    break;
  }
  case TokenKind::SquareClose: {
    kindString = "]";
    break;
  }
  case TokenKind::Equal: {
    kindString = "=";
    break;
  }
  }

  return kindString;
}

} // namespace rust_compiler::toml
