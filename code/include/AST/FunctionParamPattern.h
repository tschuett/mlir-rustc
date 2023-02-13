#pragma once

#include "AST/AST.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Types/Types.h"

#include <memory>

namespace rust_compiler::ast {

class FunctionParamPattern : public Node {
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> name;

public:
  FunctionParamPattern(Location loc) : Node(loc) {}

  void setName(std::shared_ptr<ast::patterns::PatternNoTopAlt> name);

  void setType(std::shared_ptr<ast::types::Type> type);

  std::shared_ptr<ast::types::Type> getType() const { return type; }
};

} // namespace rust_compiler::ast
