#pragma once

#include "Toml/KeyValuePair.h"
#include <string>
#include <vector>

namespace rust_compiler::toml {

class Table {
  std::string header;
  std::vector<KeyValuePair> kvs;

public:
  void setHeader(std::string_view header);
  void addPair(KeyValuePair &pair);
};

} // namespace rust_compiler::toml
