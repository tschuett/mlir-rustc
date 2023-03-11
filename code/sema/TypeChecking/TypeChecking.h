#pragma once

#include "AST/Crate.h"

#include <memory>

namespace rust_compiler::sema::type_checking {

/// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html
class TypeCheckContext {
public:
  static TypeCheckContext *get();

  void checkCrate(std::shared_ptr<ast::Crate>);
};

} // namespace rust_compiler::sema::type_checking
