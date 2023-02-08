#pragma once

#include "AST/AST.h"
#include "AST/OuterAttribute.h"
#include "AST/StructField.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/Types.h"
#include "AST/Visiblity.h"

#include <optional>
#include <string>
#include <vector>

namespace rust_compiler::ast {

class StructField : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::optional<Visibility> visiblity;
  std::string identifier;
  std::shared_ptr<ast::types::TypeExpression> type;

public:
  StructField(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast
