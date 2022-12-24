#include "Toml/Table.h"

#include "Toml/KeyValuePair.h"

namespace rust_compiler::toml {

void Table::setHeader(std::string_view _header) { header = _header; }

void Table::addPair(std::shared_ptr<KeyValuePair> pair) { kvs.push_back(pair); }

std::string Table::toString() {
  assert(false);

  return std::string("");
};

size_t Table::getNrOfTokens() {
  size_t sum = 3; // header

  for (unsigned i = 0; i < kvs.size(); ++i) {
    sum += kvs[i]->getNrOfTokens();
  }

  return sum;
};

} // namespace rust_compiler::toml
