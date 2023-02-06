#pragma once

#include "AST/OuterAttribute.h"
#include "Location.h"

#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

class ClippyAttribute : public OuterAttribute {
  unsigned lintTokens;
  std::vector<std::string> lints;

public:
  ClippyAttribute(rust_compiler::Location location,
                  std::span<std::string> _lints, unsigned lintTokens)
    : OuterAttribute(location, OuterAttributeKind::ClippyAttribute), lintTokens(lintTokens) {
    lints = {_lints.begin(), _lints.end()};
  }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
