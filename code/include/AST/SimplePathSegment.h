#pragma once

#include "AST/AST.h"

#include <cstddef>
#include <string_view>

namespace rust_compiler::ast {

class SimplePathSegment : public Node {
public:
  SimplePathSegment(std::string_view segment);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
