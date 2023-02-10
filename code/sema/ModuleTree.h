#pragma once

#include "AST/Module.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::sema {

class Module {
  std::vector<std::shared_ptr<ast::Module>> childs;

public:
  Module(std::string_view name);

  void addModule(std::shared_ptr<ast::Module> child);
};

} // namespace rust_compiler::sema

// FIXME imports
