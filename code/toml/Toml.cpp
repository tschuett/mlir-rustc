#include "Toml/Toml.h"

#include "Toml/KeyValuePair.h"
#include "Toml/Lexer.h"
#include "Toml/Parser.h"

#include <optional>

namespace rust_compiler::toml {

void Toml::addTable(std::shared_ptr<Table> tab) { tables.push_back(tab); }

void Toml::addKeyValuePair(std::shared_ptr<KeyValuePair> kv) {
  kvs.push_back(kv);
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
