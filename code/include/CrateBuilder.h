#pragma once

#include "AST/Module.h"

#include <string_view>

namespace rust_compiler {


class CrateBuilder {
public:
  CrateBuilder(std::string_view moduleName);

  void build(std::shared_ptr<ast::Module> m);
};

} // namespace rust_compiler
