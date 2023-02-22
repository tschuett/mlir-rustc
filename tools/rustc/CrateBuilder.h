#pragma once

#include "Basic/Edition.h"
#include "Basic/Ids.h"

#include <string_view>

#include "CrateLoader/CrateLoader.h"

namespace rust_compiler::rustc {

extern void buildCrate(std::string_view path, std::string_view crateName,
                       basic::CrateNum crateNum, basic::Edition edition,
                       rust_compiler::crate_loader::LoadMode mode);

}
