#pragma once

#include <string_view>

namespace rust_compiler::basic {

enum class Edition { Edition2015, Edition2018, Edition2021, Edition2024 };

 Edition stringToEdition(std::string_view edition);

}
