#include "Lexer/Token.h"

#include "Lexer/KeyWords.h"

namespace rust_compiler::lexer {

bool Token::isUseToken() const {
  return kind == TokenKind::Keyword && storage == "use";
}

bool Token::isPubToken() const {
  return kind == TokenKind::Keyword && storage == "pub";
}

bool Token::isIdentifier() const { return kind == TokenKind::Identifier; }

bool Token::isAs() const {
  return kind == TokenKind::Keyword && storage == "as";
}

std::string Token2String(TokenKind kind) {
  switch (kind) {
  case TokenKind::Colon: {
    return std::string(":");
  }
  case TokenKind::RArrow: {
    return std::string("->");
  }
    //  case TokenKind::LessThan: {
    //    return std::string("<");
    //  }
    //  case TokenKind::DoubleGreaterThan: {
    //    return std::string(">>");
    //  }
  case TokenKind::Gt: {
    return std::string(">");
  }
  case TokenKind::EqEq: {
    return std::string("==");
  }
  case TokenKind::Eq: {
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
  case TokenKind::Keyword: {
    return std::string("keyword");
  }
  case TokenKind::ParenOpen: {
    return std::string("(");
  }
  case TokenKind::ParenClose: {
    return std::string(")");
  }
  case TokenKind::Semi: {
    return std::string(";");
  }
  case TokenKind::BraceOpen: {
    return std::string("{");
  }
  case TokenKind::BraceClose: {
    return std::string("}");
  }
    //  case TokenKind::Amp: {
    //    return std::string("&");
    //  }
  case TokenKind::AndAnd: {
    return std::string("&&");
  }
    //  case TokenKind::String: {
    //    return std::string("string");
    //  }
    //  case TokenKind::Pipe: {
    //    return std::string("|");
    //  }
  case TokenKind::Star: {
    return std::string("*");
  }
    //  case TokenKind::Dash: {
    //    return std::string("-");
    //  }
    //  case TokenKind::Char: {
    //    return std::string("char");
    //  }
  case TokenKind::Plus: {
    return std::string("+");
  }
  case TokenKind::Not: {
    return std::string("!");
  }
  case TokenKind::Minus: {
    return std::string("-");
  }
  case TokenKind::Slash: {
    return std::string("/");
  }
  case TokenKind::Percent: {
    return std::string("%");
  }
  case TokenKind::And: {
    return std::string("&");
  }
  case TokenKind::Or: {
    return std::string("|");
  }
  case TokenKind::Caret: {
    return std::string("^");
  }
  case TokenKind::Shl: {
    return std::string("<<");
  }
  case TokenKind::Shr: {
    return std::string(">>");
  }
  case TokenKind::DotDot: {
    return std::string("..");
  }
  case TokenKind::Lt: {
    return std::string("<");
  }
  case TokenKind::OrOr: {
    return std::string("||");
  }
  case TokenKind::PathSep: {
    return std::string("::");
  }
  case TokenKind::DotDotDot: {
    return std::string("...");
  }
  case TokenKind::DotDotEq: {
    return std::string("..=");
  }
  case TokenKind::Ne: {
    return std::string("!=");
  }
  case TokenKind::Ge: {
    return std::string(">=");
  }
  case TokenKind::Le: {
    return std::string("<=");
  }
  case TokenKind::Underscore: {
    return std::string("_");
  }
  case TokenKind::LIFETIME_TOKEN: {
    return std::string("lifetime token");
  }
  case TokenKind::LIFETIME_OR_LABEL: {
    return std::string("lifetime or label");
  }
    //  case TokenKind::ASCII_ESCAPE: {
    //    return std::string("ascii escape");
    //  }
    //  case TokenKind::QUOTE_ESCAPE: {
    //    return std::string("quote escape");
    //  }
  case TokenKind::CHAR_LITERAL: {
    return std::string("char literal");
  }
  case TokenKind::STRING_LITERAL: {
    return std::string("string literal");
  }
  case TokenKind::RAW_STRING_LITERAL: {
    return std::string("raw string literal");
  }
  case TokenKind::BYTE_STRING_LITERAL: {
    return std::string("byte string literal");
  }
  case TokenKind::RAW_BYTE_STRING_LITERAL: {
    return std::string("raw byte string literal");
  }
  case TokenKind::BYTE_LITERAL: {
    return std::string("byte literal");
  }
  case TokenKind::FLOAT_LITERAL: {
    return std::string("float literal");
  }
  case TokenKind::Eof: {
    return std::string("eof");
  }
  case TokenKind::At: {
    return std::string("@");
  }
    //  case TokenKind::RESERVED_NUMBER: {
    //    return std::string("reserved number");
    //  }
  case TokenKind::StarEq: {
    return std::string("*=");
  }
  case TokenKind::PlusEq: {
    return std::string("+=");
  }
  case TokenKind::MinusEq: {
    return std::string("-=");
  }
  case TokenKind::AndEq: {
    return std::string("&=");
  }
  case TokenKind::ShlEq: {
    return std::string("<<=");
  }
  case TokenKind::ShrEq: {
    return std::string(">>=");
  }
  case TokenKind::OrEq: {
    return std::string("|=");
  }
  case TokenKind::PercentEq: {
    return std::string("%=");
  }
    //  case TokenKind::UNICODE_ESCAPE: {
    //    return std::string("unicode escape");
    //  }
  case TokenKind::CaretEq: {
    return std::string("^=");
  }
  case TokenKind::INTEGER_LITERAL: {
    return std::string("integer literal");
  }
  case TokenKind::FatArrow: {
    return std::string("=>");
  }
  case TokenKind::SlashEq: {
    return std::string("/=");
  }
  case TokenKind::Dollar: {
    return std::string("?");
  }
  case TokenKind::Tilde: {
    return std::string("~");
  }
  }
}

std::string Token::toString() {
  switch (kind) {
  case TokenKind::Colon:
    return ":";
  case TokenKind::Eq:
    return "=";
  case TokenKind::Dot:
    return ".";
  case TokenKind::QMark:
    return "?";
  case TokenKind::SquareOpen:
    return "[";
  case TokenKind::SquareClose:
    return "]";
  case TokenKind::ParenOpen:
    return "(";
  case TokenKind::ParenClose:
    return ")";
  case TokenKind::BraceOpen:
    return "{";
  case TokenKind::BraceClose:
    return "}";
  case TokenKind::Hash:
    return "#";
  case TokenKind::Comma:
    return ",";
  case TokenKind::Semi:
    return ";";
  case TokenKind::Star:
    return "*";
  case TokenKind::StarEq:
    return "*=";
  case TokenKind::Tilde:
    return "~";
  case TokenKind::Dollar:
    return "?";
  case TokenKind::Plus:
    return "+";
  case TokenKind::PlusEq:
    return "+=";
  case TokenKind::Minus:
    return "-";
  case TokenKind::MinusEq:
    return "-=";
  case TokenKind::RArrow:
    return "->";
  case TokenKind::FatArrow:
    return "=>";
  case TokenKind::Not:
    return "!";
  case TokenKind::Percent:
    return "%";
  case TokenKind::Slash:
    return "/";
  case TokenKind::And:
    return "&";
  case TokenKind::AndEq:
    return "&=";
  case TokenKind::AndAnd:
    return "&&";
  case TokenKind::Or:
    return "|";
  case TokenKind::OrEq:
    return "|=";
  case TokenKind::OrOr:
    return "||";
  case TokenKind::PathSep:
    return "::";
  case TokenKind::Caret:
    return "^";
  case TokenKind::ShrEq:
    return ">>=";
  case TokenKind::Shr:
    return ">>";
  case TokenKind::Shl:
    return "<<";
  case TokenKind::ShlEq:
    return "<<=";
  case TokenKind::DotDot:
    return "..";
  case TokenKind::DotDotDot:
    return "...";
  case TokenKind::DotDotEq:
    return "..=";
  case TokenKind::Lt:
    return "<";
  case TokenKind::EqEq:
    return "==";
  case TokenKind::Ne:
    return "!=";
  case TokenKind::Gt:
    return ">";
  case TokenKind::Ge:
    return ">=";
  case TokenKind::Le:
    return "<=";
  case TokenKind::Underscore:
    return "_";
  case TokenKind::SlashEq:
    return "/=";
  case TokenKind::PercentEq:
    return "%=";
  case TokenKind::CaretEq:
    return "^=";
  case TokenKind::At:
    return "@";
  case TokenKind::Identifier:
    return ident.toString();
  case TokenKind::Keyword:
    return *KeyWord2String(kw);
  case TokenKind::INTEGER_LITERAL:
    return storage;
  case TokenKind::FLOAT_LITERAL:
    return storage;
  case TokenKind::CHAR_LITERAL:
    return charToString();
  case TokenKind::BYTE_LITERAL:
    return storage;
  case TokenKind::RAW_BYTE_STRING_LITERAL:
    return storage;
  case TokenKind::LIFETIME_OR_LABEL:
    return LifetimeOrLabelToString();
  case TokenKind::LIFETIME_TOKEN:
    return LifetimeTokenToString();
  case TokenKind::STRING_LITERAL:
    return utf8Storage.toString();
  case TokenKind::RAW_STRING_LITERAL:
    return utf8Storage.toString();
  case TokenKind::BYTE_STRING_LITERAL:
    return storage;
  case TokenKind::Eof:
    return "";
  }
}

std::string Token::charToString() const { return utf8Storage.toString(); }

} // namespace rust_compiler::lexer
