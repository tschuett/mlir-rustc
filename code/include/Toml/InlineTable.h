#pragma once

#include "Toml/KeyValuePair.h"

#include <memory>
#include <string>
#include <vector>

namespace rust_compiler::toml {

class InlineTable : public Value {
  std::vector<std::shared_ptr<KeyValuePair>> kvs;

public:
  void addPair(std::shared_ptr<KeyValuePair> pair);

  size_t getNrOfTokens() override;
  std::string toString() override;
};

} // namespace rust_compiler::toml
