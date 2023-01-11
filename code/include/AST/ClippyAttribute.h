#pragma once

#include "AST/Item.h"
#include "AST/Statement.h"
#include "Location.h"

#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

class ClippyAttribute : public Item {
  unsigned lintTokens;
  std::vector<std::string> lints;

public:
  ClippyAttribute(rust_compiler::Location location,
                  std::span<std::string> _lints, unsigned lintTokens)
    : Item(location, ItemKind::ClippyAttribute), lintTokens(lintTokens) {
    lints = {_lints.begin(), _lints.end()};
  }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
