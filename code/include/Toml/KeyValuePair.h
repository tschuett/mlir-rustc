#pragma once

#include "Toml/Array.h"

#include <string>
#include <string_view>
#include <variant>

namespace rust_compiler::toml {

class KeyValuePair {
  std::string key;
  std::variant<std::string, Array> value;

public:
  KeyValuePair(std::string_view key, std::string_view value)
    : key(key), value(std::string(value)) {}

  size_t getNrOfTokens();
};

} // namespace rust_compiler::toml
