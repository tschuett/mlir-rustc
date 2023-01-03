#pragma once

#include "AST/AST.h"

#include <cstddef>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class SimplePathSegment : public Node {
  std::string segment;

public:
  SimplePathSegment(std::string_view segment) : segment(segment){};

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
