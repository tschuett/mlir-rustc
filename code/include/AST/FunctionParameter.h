#pragma once

#include "AST/AST.h"
#include "AST/PatternNoTopAlt.h"
#include "AST/Types/Types.h"

#include <memory>

namespace rust_compiler::ast {

class FunctionParameter : public Node {
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::PatternNoTopAlt> name;

public:
  FunctionParameter(Location loc) : Node(loc) {}

  void setName(std::shared_ptr<ast::PatternNoTopAlt> name);

  void setType(std::shared_ptr<ast::types::Type> type);

  std::shared_ptr<ast::types::Type> getType() const { return type; }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
