#pragma once

#include "AST/Crate.h"

#include <memory>

namespace rust_compiler::sema::type_checking {

class TypeCheckContext {
public:
  static TypeCheckContext *get();

  void resolveCrate(std::shared_ptr<ast::Crate>);
};

} // namespace rust_compiler::sema::type_checking
