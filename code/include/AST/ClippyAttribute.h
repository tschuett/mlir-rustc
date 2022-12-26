#pragma once

#include "AST/Item.h"

#include <span>
#include <string>
#include <vector>

namespace rust_compiler::ast {

class ClippyAttribute : public Item {
  std::vector<std::string> lints;

public:
  ClippyAttribute(std::span<std::string> lints);
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
