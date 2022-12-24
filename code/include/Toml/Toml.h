#pragma once

#include "Toml/KeyValuePair.h"
#include "Toml/Table.h"

#include <optional>
#include <string_view>

namespace rust_compiler::toml {

// https://toml.io/en/

class Toml {
  std::vector<std::shared_ptr<KeyValuePair>> kvs;
  std::vector<std::shared_ptr<Table>> tables;

public:
  void addTable(std::shared_ptr<Table> tab);
  void addKeyValuePair(std::shared_ptr<KeyValuePair> kv);

  std::optional<std::string> getEdition();
};

extern std::optional<Toml> readToml(std::string_view file);

} // namespace rust_compiler::toml
