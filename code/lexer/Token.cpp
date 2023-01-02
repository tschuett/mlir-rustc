#include "Lexer/Token.h"

namespace rust_compiler::lexer {

std::string Token2String(TokenKind kind) {

  switch (kind) {
  case TokenKind::DoubleColon: {
    return std::string("::");
  }
  case TokenKind::Colon: {
    return std::string(":");
  }
  case TokenKind::ThinArrow: {
    return std::string("->");
  }
  case TokenKind::LessThan: {
    return std::string("<");
  }
  case TokenKind::DoubleGreaterThan: {
    return std::string(">>");
  }
  case TokenKind::GreaterThan: {
    return std::string(">");
  }
  case TokenKind::Equals: {
    return std::string("=");
  }
  case TokenKind::Dot: {
    return std::string(".");
  }
  case TokenKind::QMark: {
    return std::string("?");
  }
  case TokenKind::Hash: {
    return std::string("#");
  }
  case TokenKind::Exclaim: {
    return std::string("!");
  }
  case TokenKind::SquareOpen: {
    return std::string("[");
  }
  case TokenKind::SquareClose: {
    return std::string("]");
  }
  case TokenKind::Comma: {
    return std::string(",");
  }
  case TokenKind::Identifier: {
    return std::string("identifier");
  }
  case TokenKind::ParenOpen: {
    return std::string("(");
  }
  case TokenKind::ParenClose: {
    return std::string(")");
  }
  case TokenKind::SemiColon: {
    return std::string(";");
  }
  case TokenKind::BraceOpen: {
    return std::string("{");
  }
  case TokenKind::BraceClose: {
    return std::string("}");
  }
  case TokenKind::Amp: {
    return std::string("&");
  }
  case TokenKind::DoubleAmp: {
    return std::string("string");
  }
  case TokenKind::String: {
    return std::string("string");
  }
  case TokenKind::Pipe: {
    return std::string("|");
  }
  case TokenKind::Star: {
    return std::string("*");
  }
  case TokenKind::Dash: {
    return std::string("-");
  }
  case TokenKind::Char: {
    return std::string("char");
  }
  case TokenKind::Plus: {
    return std::string("+");
  }
  }
}

} // namespace rust_compiler::lexer
