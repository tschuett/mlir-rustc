#include "Toml/Table.h"
#include "Toml/KeyValuePair.h"

namespace rust_compiler::toml {

void Table::setHeader(std::string_view _header) { header = _header; }

void Table::addPair(KeyValuePair &pair) {
  kvs.push_back(pair);
}

} // namespace rust_compiler::toml
