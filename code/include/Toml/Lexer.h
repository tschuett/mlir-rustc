#pragma once

#include "Toml/TokenStream.h"

#include <string_view>
#include <optional>

namespace rust_compiler::toml {

std::optional<TokenStream> tryLexToml(std::string_view toml);

}
