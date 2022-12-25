#pragma once

#include "TokenStream.h"

#include <string_view>

namespace rust_compiler {

  void parser(TokenStream& ts, std::string_view path);
}
