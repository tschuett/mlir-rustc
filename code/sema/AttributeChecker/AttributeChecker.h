#pragma once

#include "AST/Crate.h"

#include <memory>

namespace rust_compiler::sema::attribute_checker {

using namespace rust_compiler::ast;

/// https://doc.rust-lang.org/reference/attributes.html

class AttributeChecker {
public:
  void checkCrate(std::shared_ptr<ast::Crate> crate);

private:
  void checkItem(Item *item);
  void checkVisItem(VisItem *item);
  void checkFunction(Function *item);
};

} // namespace rust_compiler::sema::attribute_checker
