#pragma once

#include "Lexer/TokenStream.h"

#include <string_view>

namespace rust_compiler::lexer {

TokenStream lex(std::string_view code, std::string_view fileName);

}
