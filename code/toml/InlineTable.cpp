#include "Toml/InlineTable.h"

#include "Toml/KeyValuePair.h"

namespace rust_compiler::toml {

void InlineTable::addPair(KeyValuePair &pair) { kvs.push_back(pair); }

size_t InlineTable::getNrOfTokens() {}

} // namespace rust_compiler::toml
