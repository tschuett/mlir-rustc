#pragma once

#include "Toml/Value.h"

#include <memory>
#include <string>
#include <variant>

class Array;

namespace rust_compiler::toml {

class KeyValuePair : public Value {
  std::string key;
  std::variant<std::shared_ptr<Value>, std::string> value;

public:
  virtual ~KeyValuePair() = default;

  KeyValuePair(std::string_view _key, std::shared_ptr<Value> _value) {
    key = _key;
    value = _value;
  }

  KeyValuePair(std::string_view _key, std::string_view _value) {
    key = _key;
    value = std::string(_value);
  }

  size_t getNrOfTokens() override;

  std::string toString() override;
};

} // namespace rust_compiler::toml
