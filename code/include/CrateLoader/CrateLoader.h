#pragma once

#include "AST/Crate.h"

#include <memory>
#include <string_view>

namespace rust_compiler::crate_loader {

enum class LoadMode { File, CargoTomlDir };

std::shared_ptr<ast::Crate> loadCrate(std::string_view path,
                                      std::string_view crateName,
                                      basic::CrateNum crateNum, LoadMode mode);

} // namespace rust_compiler::crate_loader
