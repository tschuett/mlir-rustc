#pragma once

#include "AST/Module.h"

#include <string_view>

namespace rust_compiler {

class ModuleBuilder {
  std::string moduleName;

public:
 ModuleBuilder(std::string_view moduleName) : moduleName(moduleName) {};

  void build(std::shared_ptr<ast::Module> m);
};

} // namespace rust_compiler
