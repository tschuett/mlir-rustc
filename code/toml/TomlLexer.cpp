#include "Toml/Lexer.h"
#include "Toml/Token.h"
#include "Toml/TokenStream.h"

#include <cstdlib>
#include <optional>
#include <string>

namespace rust_compiler::toml {

static std::optional<std::string> tryLexKey(std::string_view str) {
  std::string key;

  while (str.size() > 1) {
    if (isalpha(str[0])) {
      str.remove_prefix(1);
      key.push_back(str[0]);
    } else if (isdigit(str[0])) {
      str.remove_prefix(1);
      key.push_back(str[0]);
    } else if (str.starts_with("_")) {
      str.remove_prefix(1);
      key.push_back(str[0]);
    } else if (str.starts_with("-")) {
      str.remove_prefix(1);
      key.push_back(str[0]);
    }
  }

  if (key.length() > 0)
    return key;

  return std::nullopt;
}

TokenStream lexToml(std::string_view toml) {
  TokenStream ts;

  while (toml.size() > 0) {

    auto key = tryLexKey(toml);
    if (key) {
      toml.remove_prefix((*key).length());
      ts.append(Token(TokenKind::Identifier, *key));
    } else {
      continue;
    }

    if (toml.starts_with("#")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::Hash));
    } else if (toml.starts_with("[")) {
      toml.remove_prefix(1);
      ts.append(Token(TokenKind::SquareOpen));
    } else {
      printf("unknown token: %s\n", toml.data());
      exit(EXIT_FAILURE);
    }
  }

  return ts;
}

} // namespace rust_compiler::toml
