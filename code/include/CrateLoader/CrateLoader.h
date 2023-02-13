#pragma once

#include "AST/Crate.h"
#include "Basic/Edition.h"

#include <memory>
#include <string_view>

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Crate> loadCrate(std::string_view path,
                                      std::string_view crateName,
                                      basic::Edition edition);

}
