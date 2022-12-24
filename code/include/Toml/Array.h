#pragma once

#include "Toml/Value.h"

#include <string>
#include <vector>

namespace rust_compiler::toml {

class Array : public Value {
  std::vector<std::string> elements;

public:
  void addElement(std::string_view element);
  size_t getNrOfTokens() override;

  std::string toString() override;
};

} // namespace rust_compiler::toml
