#include "Toml/KeyValuePair.h"

#include "Toml/InlineTable.h"

#include <sstream>
#include <variant>

namespace rust_compiler::toml {

size_t KeyValuePair::getNrOfTokens() {

  if (std::holds_alternative<std::shared_ptr<Value>>(value)) {
    return std::get<std::shared_ptr<Value>>(value)->getNrOfTokens() + 2;
  } else {
    return 3;
  }
}

std::string KeyValuePair::toString() {

  std::stringstream s;

  s << key << ": ";

  if (std::holds_alternative<std::shared_ptr<Value>>(value)) {
    s << std::get<std::shared_ptr<Value>>(value)->toString();
  } else {
    s << std::get<std::string>(value);
  }

  return s.str();
}

} // namespace rust_compiler::toml
