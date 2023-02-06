#pragma once

#include "AST/AST.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/Types.h"

#include <vector>
#include <memory>
#include <optional>

namespace rust_compiler::ast {

class ClosureParam : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<patterns::PatternNoTopAlt> pattern;
  std::optional<std::shared_ptr<types::Type>> type;

public:
  ClosureParam(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
