#include "Toml/Toml.h"

#include "Toml/KeyValuePair.h"
#include "Toml/Lexer.h"
#include "Toml/Parser.h"

#include <optional>
#include <variant>

namespace rust_compiler::toml {

void Toml::addTable(std::shared_ptr<Table> tab) { tables.push_back(tab); }

void Toml::addKeyValuePair(std::shared_ptr<KeyValuePair> kv) {
  kvs.push_back(kv);
}

std::optional<std::string> Toml::getEdition() {
  for (auto kv : kvs) {
    if (kv->getKey() == "edition") {
      return kv->getStringVariant();
    }
  }

  return std::nullopt;
}

std::optional<Toml> readToml(std::string_view file) {

  std::optional<TokenStream> ts = tryLexToml(file);
  if (!ts)
    return std::nullopt;

  printf("lexed file\n");

  std::optional<Toml> toml = tryParse(*ts);

  if (toml)
    return *toml;

  return std::nullopt;
}

} // namespace rust_compiler::toml
