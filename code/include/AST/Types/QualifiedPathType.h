#pragma once

#include "AST/AST.h"
#include "AST/Types/TypePath.h"
#include "Location.h"

#include <memory>

namespace rust_compiler::ast::types {

class QualifiedPathType final : public Node {
  std::shared_ptr<ast::types::TypeExpression> type;
  std::shared_ptr<ast::types::TypeExpression> path;

public:
  QualifiedPathType(Location loc) : Node(loc) {}

  void setType(std::shared_ptr<ast::types::TypeExpression>);
  void setPath(std::shared_ptr<ast::types::TypeExpression>);
};

} // namespace rust_compiler::ast::types
