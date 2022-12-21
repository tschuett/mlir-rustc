#pragma once

#include "Toml/Toml.h"
#include "Toml/TokenStream.h"

namespace rust_compiler::toml {

std::optional<Toml> tryParse(TokenStream ts);

}
