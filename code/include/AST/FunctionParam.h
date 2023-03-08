#pragma once

#include "AST/AST.h"
#include "AST/FunctionParamPattern.h"
#include "AST/OuterAttribute.h"
#include "AST/Patterns/IdentifierPattern.h"
#include "AST/Types/TypeExpression.h"
#include "AST/VariableDeclaration.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast {

enum class FunctionParamKind { Pattern, DotDotDot, Type };

class FunctionParam : public Node {
  FunctionParamKind kind;
  std::vector<ast::OuterAttribute> outerAttributes;
  std::shared_ptr<ast::types::TypeExpression> type;
  FunctionParamPattern pattern;

public:
  FunctionParam(Location loc, FunctionParamKind kind)
      : Node(loc), kind(kind), pattern(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> out) {
    outerAttributes = {out.begin(), out.end()};
  }
  void setType(std::shared_ptr<ast::types::TypeExpression> type);

  std::shared_ptr<ast::types::TypeExpression> getType() const { return type; }

  FunctionParamKind getKind() const { return kind; }

  void setPattern(const FunctionParamPattern &pat) { pattern = pat; }

  FunctionParamPattern getPattern() const { return pattern; }
};

} // namespace rust_compiler::ast
