#pragma once

#include "TokenStream.h"

#include <string_view>

namespace rust_compiler {

TokenStream lex(std::string_view code);

}
