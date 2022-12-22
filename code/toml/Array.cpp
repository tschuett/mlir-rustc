#include "Toml/Array.h"

namespace rust_compiler::toml {

void Array::addElement(std::string_view element) {
  elements.push_back(std::string(element));
}

size_t Array::getNrOfTokens() { return elements.size() + elements.size() - 1; }

} // namespace rust_compiler::toml
