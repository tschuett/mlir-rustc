#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/Types.h"
#include "AST/Visiblity.h"

#include <optional>
#include <vector>
#include <memory>

namespace rust_compiler::ast {

class TupleField : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  std::shared_ptr<ast::types::Type> type;

public:
  TupleField(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
