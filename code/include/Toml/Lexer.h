#pragma once

#include "Toml/TokenStream.h"

#include <string_view>

namespace rust_compiler::toml {

TokenStream lexToml(std::string_view toml);

}
