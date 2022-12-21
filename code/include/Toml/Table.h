#pragma once

#include <string>
#include <vector>

namespace rust_compiler::toml {

class Table {
  std::string header;
  std::vector<std::pair<std::string, std::string>> kvs;

public:
  void setHeader(std::string_view header);
  void addPair(std::pair<std::string, std::string> &pair);
};

} // namespace rust_compiler::toml
