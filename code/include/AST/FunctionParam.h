#pragma once

#include "AST/AST.h"
#include "AST/FunctionParamPattern.h"

#include <memory>

namespace rust_compiler::ast {

class FunctionParam : public Node {
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> name;

public:
  FunctionParam(Location loc) : Node(loc) {}

  void setName(std::shared_ptr<ast::patterns::PatternNoTopAlt> name);

  void setType(std::shared_ptr<ast::types::Type> type);

  std::shared_ptr<ast::types::Type> getType() const { return type; }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
