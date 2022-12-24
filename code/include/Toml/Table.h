#pragma once

#include "Toml/KeyValuePair.h"
#include "Toml/Value.h"

#include <string>
#include <vector>

namespace rust_compiler::toml {

class Table : public Value {
  std::string header;
  std::vector<std::shared_ptr<KeyValuePair>> kvs;

public:
  virtual ~Table() = default;

  void setHeader(std::string_view header);
  void addPair(std::shared_ptr<KeyValuePair> pair);

  std::string toString() override;
  size_t getNrOfTokens() override;
};

} // namespace rust_compiler::toml
