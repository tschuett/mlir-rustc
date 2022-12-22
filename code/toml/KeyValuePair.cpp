#include "Toml/KeyValuePair.h"

#include <variant>

namespace rust_compiler::toml {

size_t KeyValuePair::getNrOfTokens() {

  if (std::holds_alternative<std::string>(value)) {
    return 3;
  } else {
    Array array = std::get<Array>(value);
    return array.getNrOfTokens();
  }
  return 0;
}

} // namespace rust_compiler::toml
