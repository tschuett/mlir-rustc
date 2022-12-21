#include "Toml/Toml.h"

#include "Toml/Lexer.h"
#include "Toml/Parser.h"

#include <optional>

namespace rust_compiler::toml {

std::optional<Toml> readToml(std::string_view file) {

  TokenStream ts = lexToml(file);

  std::optional<Toml> toml = tryParse(ts);

  return std::nullopt;
}

} // namespace rust_compiler::toml
