#include "Toml/Array.h"

#include <sstream>

namespace rust_compiler::toml {

void Array::addElement(std::string_view element) {
  elements.push_back(std::string(element));
}

size_t Array::getNrOfTokens() {
  return elements.size() + elements.size() - 1 + 2;
}

std::string Array::toString() {
  std::stringstream s;

  for (unsigned i = 0; i < elements.size(); ++i) {
    s << elements[i];
    if (i + 1 != elements.size()) {
      s << ",";
    }
  }

  return s.str();
}

} // namespace rust_compiler::toml
