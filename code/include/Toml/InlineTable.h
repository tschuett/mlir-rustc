#pragma once

#include "Toml/KeyValuePair.h"

#include <string>
#include <vector>

namespace rust_compiler::toml {

class InlineTable {
  std::vector<KeyValuePair> kvs;

public:
  void addPair(KeyValuePair &pair);

  size_t getNrOfTokens();
};

} // namespace rust_compiler::toml
