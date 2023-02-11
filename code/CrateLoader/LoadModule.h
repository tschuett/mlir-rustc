#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/Module.h"

#include <memory>
#include <string_view>

namespace rust_compiler::crate_loader {

std::shared_ptr<ast::Module>
loadRootModule(llvm::SmallVectorImpl<char> &libPath, std::string_view crateName,
               adt::CanonicalPath canonicalPath);

std::shared_ptr<ast::Module> loadModule(llvm::SmallVectorImpl<char> &libPath,
                                        std::string_view fileName,
                                        std::string_view crateName,
                                        adt::CanonicalPath canonicalPath);

} // namespace rust_compiler::crate_loader
