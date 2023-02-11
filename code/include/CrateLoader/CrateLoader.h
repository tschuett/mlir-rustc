#pragma once

#include "AST/Crate.h"

#include <memory>
#include <string_view>

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Crate> loadCrate(std::string_view path,
                                      std::string_view crateName,
                                      std::string_view edition);

}
