#pragma once

#include <cstddef>
#include <string>

namespace rust_compiler::toml {

class Value {
public:
  virtual ~Value() = default;

  virtual std::string toString() = 0;
  virtual size_t getNrOfTokens() = 0;
};

} // namespace rust_compiler::toml
