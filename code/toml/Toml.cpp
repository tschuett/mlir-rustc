#include "Toml/Toml.h"

#include "Toml/Lexer.h"

#include <optional>

namespace rust_compiler::toml {

std::optional<Toml> readToml(std::string_view file) {

  lexToml(file);

  return std::nullopt;
}

} // namespace rust_compiler::toml
