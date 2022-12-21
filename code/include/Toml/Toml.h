#pragma once

#include "Toml/Table.h"

#include <optional>
#include <string_view>

namespace rust_compiler::toml {

// https://toml.io/en/

class Toml {
  std::vector<std::pair<std::string, std::string>> kvs;
  std::vector<Table> tables;
public:
  void addTable(Table &tab);
  void addKeyValuePair(const std::pair<std::string, std::string>& kv);
};

extern std::optional<Toml> readToml(std::string_view file);

} // namespace rust_compiler::toml
