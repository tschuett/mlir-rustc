#pragma once

#include <string_view>

namespace rust_compiler::rustc {

extern void buildCrate(std::string_view path, std::string_view edition);

}
