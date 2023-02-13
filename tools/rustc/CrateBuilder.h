#pragma once

#include <string_view>

#include "Basic/Edition.h"

namespace rust_compiler::rustc {

extern void buildCrate(std::string_view path, std::string_view crateName,
                       basic::Edition edition);

}
