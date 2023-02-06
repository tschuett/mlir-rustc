#pragma once

#include "AST/AST.h"
#include "AST/Attr.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

class InnerAttribute : public Node {
  Attr attr;

public:
  InnerAttribute(rust_compiler::Location location)
      : Node(location), attr(location) {}

  SimplePath getPath() const;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
