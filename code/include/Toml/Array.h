#pragma once

#include <string>
#include <vector>

namespace rust_compiler::toml {

class Array {
  std::vector<std::string> elements;

public:
  void addElement(std::string_view element);
  size_t getNrOfTokens();
};

} // namespace rust_compiler::toml
