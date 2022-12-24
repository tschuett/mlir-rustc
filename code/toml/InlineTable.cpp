#include "Toml/InlineTable.h"

#include "Toml/KeyValuePair.h"

#include <sstream>

namespace rust_compiler::toml {

void InlineTable::addPair(std::shared_ptr<KeyValuePair> pair) {
  kvs.push_back(pair);
}

size_t InlineTable::getNrOfTokens() {
  size_t sum = 2; // {}

  for (unsigned i = 0; i < kvs.size(); ++i) {
    sum += kvs[i]->getNrOfTokens();
    if (i + 1 < kvs.size()){
      sum += 1; // comma
    }
  }

  return sum;
}

std::string InlineTable::toString() {

  std::stringstream s;

  s << std::string("{");

  for (unsigned i = 0; i < kvs.size(); ++i) {
    s << kvs[i]->toString();
    if (i + 1 != kvs.size())
      s << ", ";
  }

  return s.str();
}

} // namespace rust_compiler::toml
