#pragma once

#include "AST/AST.h"
#include "AST/FunctionParamPattern.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Types/TypeExpression.h"
#include "AST/VariableDeclaration.h"
#include "AST/OuterAttribute.h"

#include <memory>
#include <optional>
#include <vector>
#include <span>

namespace rust_compiler::ast {

enum class FunctionParamKind { Pattern, DotDotDot, Type };

class FunctionParam : public VariableDeclaration {
  FunctionParamKind kind;
  std::vector<ast::OuterAttribute> outerAttributes;
  std::shared_ptr<ast::types::TypeExpression> type;
  std::shared_ptr<ast::patterns::IdentifierPattern> name;

public:
  FunctionParam(Location loc, FunctionParamKind kind)
    : VariableDeclaration(loc, VariableDeclarationKind::FunctionParameter), kind(kind) {}

  void setAttributes(std::span<OuterAttribute>);
  void setName(std::shared_ptr<ast::patterns::IdentifierPattern> name);
  void setType(std::shared_ptr<ast::types::TypeExpression> type);

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }

  std::string getName() override;
};

} // namespace rust_compiler::ast
