#include "Toml/Lexer.h"
#include "Toml/Token.h"
#include "Toml/TokenStream.h"

#include <cstdlib>
#include <optional>
#include <string>

namespace rust_compiler::toml {

static std::optional<std::string> tryLexString(std::string_view str) {
  std::string key;

  if (not str.starts_with("\""))
    return std::nullopt;

  key.push_back(str[0]);
  str.remove_prefix(1);

  while (str.size() > 1) {
    if (str.starts_with("\"")) {
      key.push_back(str[0]);
      str.remove_prefix(1);
      return key;
    } else if (isdigit(str[0])) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else if (isascii(str[0])) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else if (str.starts_with("_")) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else {
      if (key.length() > 0)
        return key;
      return std::nullopt;
    }
  }

  if (key.length() > 0)
    return key;

  return std::nullopt;
}

static std::optional<std::string> tryLexInteger(std::string_view str) {
  std::string key;

  while (str.size() > 1) {
    if (isdigit(str[0])) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else {
      if (key.length() > 0)
        return key;
      return std::nullopt;
    }
  }

  if (key.length() > 0)
    return key;

  return std::nullopt;
}

static std::optional<std::string> tryLexKey(std::string_view str) {
  std::string key;

  while (str.size() > 1) {
    if (isalpha(str[0])) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else if (isdigit(str[0])) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else if (str.starts_with("_")) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else if (str.starts_with("-")) {
      key.push_back(str[0]);
      str.remove_prefix(1);
    } else {
      if (key.length() > 0)
        return key;
      return std::nullopt;
    }
  }

  if (key.length() > 0)
    return key;

  return std::nullopt;
}

static std::string consumeRestOfLine(std::string_view line) {
  std::string rest;

  while (line.size() > 0) {
    if (line.starts_with("\n")) {
      line.remove_prefix(1);
      rest.push_back(line[0]); // \n
      return rest;
    } else {
      rest.push_back(line[0]);
      line.remove_prefix(1);
    }
  }

  return rest;
}

std::optional<TokenStream> tryLexToml(std::string_view toml) {
  TokenStream ts;

  while (toml.size() > 0) {

    auto key = tryLexKey(toml);
    if (key) {
      toml.remove_prefix((*key).length());
      ts.append(Token(TokenKind::Identifier, *key));
    }

    auto str = tryLexString(toml);
    if (str) {
      toml.remove_prefix((*str).length());
      ts.append(Token(TokenKind::String, *str));
    }

    auto integer = tryLexInteger(toml);
    if (integer) {
      toml.remove_prefix((*integer).length());
      ts.append(Token(TokenKind::Integer, *integer));
    }

    if (toml.starts_with("#")) {
      toml.remove_prefix(1);
      std::string rest = consumeRestOfLine(toml);
      toml.remove_prefix(rest.length());
      //ts.append(Token(TokenKind::Hash));
    } else if (toml.starts_with("[")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::SquareOpen));
    } else if (toml.starts_with("]")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::SquareClose));
    } else if (toml.starts_with("=")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::Equal));
    } else if (toml.starts_with("\n")) {
      toml.remove_prefix(1);
    } else if (toml.starts_with("{")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::BraceOpen));
    } else if (toml.starts_with("}")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::BraceClose));
    } else if (toml.starts_with(",")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::Comma));
    } else if (toml.starts_with("\t")) {
      toml.remove_prefix(1);
    } else if (toml.starts_with(" ")) {
      toml.remove_prefix(1);
    } else {
      printf("unknown token: %s\n", toml.data());
      exit(EXIT_FAILURE);
    }
  }

  return ts;
}

} // namespace rust_compiler::toml
