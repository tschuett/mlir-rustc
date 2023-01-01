#pragma once

#include "AST/Item.h"
#include "AST/Statement.h"
#include <mlir/IR/Location.h>

#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

class ClippyAttribute : public Item {
  std::vector<std::string> lints;

public:
  ClippyAttribute(mlir::Location location, std::span<std::string> _lints)
      : Item{location} {
    lints = {_lints.begin(), _lints.end()};
  }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
