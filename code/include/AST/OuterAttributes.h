#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class OuterAttributes : public Node {
  std::vector<std::shared_ptr<OuterAttribute>> outerAttributes;

public:
  OuterAttributes(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
