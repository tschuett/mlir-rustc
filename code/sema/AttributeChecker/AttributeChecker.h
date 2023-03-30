#pragma once

#include "AST/Crate.h"

#include <memory>

namespace rust_compiler::sema::attribute_checker {

/// https://doc.rust-lang.org/reference/attributes.html

class AttributeChecker {
public:
  void checkCrate(std::shared_ptr<ast::Crate> crate);
};

} // namespace rust_compiler::sema::attribute_checker
