#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/Expression.h"

#include <memory>
#include <variant>

namespace rust_compiler::ast {

enum class AttrInputKind { DelimTokenTree, Expression };

class AttrInput : public Node {
  std::variant<DelimTokenTree, std::shared_ptr<Expression>> input;

public:
  AttrInput(Location loc) : Node(loc) {}

  AttrInputKind getKind() const;

  void setTokenTree(const DelimTokenTree &tree) { input = tree; }
  void setExpression(std::shared_ptr<Expression> expr) { input = expr; }
};

} // namespace rust_compiler::ast
