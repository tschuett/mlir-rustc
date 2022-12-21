#include "Toml/Table.h"

namespace rust_compiler::toml {

void Table::setHeader(std::string_view _header) { header = _header; }

void Table::addPair(std::pair<std::string, std::string> &pair) {
  kvs.push_back(pair);
}

} // namespace rust_compiler::toml
