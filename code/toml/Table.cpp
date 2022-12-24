#include "Toml/Table.h"

#include "Toml/KeyValuePair.h"

namespace rust_compiler::toml {

void Table::setHeader(std::string_view _header) { header = _header; }

void Table::addPair(std::shared_ptr<KeyValuePair> pair) {
  kvs.push_back(pair);
}

std::string Table::toString() {
  assert(false);

  return std::string("");
};

size_t Table::getNrOfTokens() {
  assert(false);
  return 0;
};

} // namespace rust_compiler::toml
