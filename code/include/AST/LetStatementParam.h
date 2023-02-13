#pragma once

#include "AST/VariableDeclaration.h"
#include "AST/Types/Types.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class LetStatementParam : public VariableDeclaration {
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<std::string> name;

public:
  LetStatementParam(Location loc)
      : VariableDeclaration(loc, VariableDeclarationKind::LetStatement) {}

  void setName(std::string name);

  void setType(std::shared_ptr<ast::types::Type> type);

  std::shared_ptr<ast::types::Type> getType() const { return type; }

  std::string getName() override;
};

} // namespace rust_compiler::ast
