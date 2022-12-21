#include "Toml/Toml.h"

#include "Toml/Lexer.h"
#include "Toml/Parser.h"

#include <optional>

namespace rust_compiler::toml {

void Toml::addKeyValuePair(const std::pair<std::string, std::string> &kv) {
  kvs.push_back(kv);
}

void Toml::addTable(Table& tab) { tables.push_back(tab); }

std::optional<Toml> readToml(std::string_view file) {

  std::optional<TokenStream> ts = tryLexToml(file);
  if (!ts)
    return std::nullopt;

  std::optional<Toml> toml = tryParse(*ts);

  if (toml)
    return *toml;

  return std::nullopt;
}

} // namespace rust_compiler::toml
