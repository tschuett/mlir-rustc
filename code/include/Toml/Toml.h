#pragma once

#include <optional>
#include <string_view>

namespace rust_compiler::toml {

// https://toml.io/en/

class Toml {};

extern std::optional<Toml> readToml(std::string_view file);

} // namespace rust_compiler::toml
