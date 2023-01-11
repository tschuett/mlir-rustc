#pragma once

#include "AST/AST.h"
#include "AST/FunctionParamPattern.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/VariableDeclaration.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class FunctionParam : public VariableDeclaration {
  std::shared_ptr<ast::types::Type> type;
  std::shared_ptr<ast::patterns::IdentifierPattern> name;

public:
  FunctionParam(Location loc)
      : VariableDeclaration(loc, VariableDeclarationKind::FunctionParameter) {}

  void setName(std::shared_ptr<ast::patterns::IdentifierPattern> name);

  void setType(std::shared_ptr<ast::types::Type> type);

  std::shared_ptr<ast::types::Type> getType() const { return type; }

  size_t getTokens() override;

  std::string getName() override;
};

} // namespace rust_compiler::ast
