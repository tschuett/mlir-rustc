#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Visiblity.h"

#include <memory>
#include <optional>
#include <vector>

namespace rust_compiler::ast {

class TupleField : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visibility;
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  TupleField(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
