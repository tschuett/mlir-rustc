#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/Expression.h"
#include "AST/SimplePath.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class MacroInvocationExpression : public ExpressionWithoutBlock {
  std::shared_ptr<ast::Expression> simplePath;
  std::shared_ptr<DelimTokenTree> tree;

  std::optional<SimplePath> path;

public:
  MacroInvocationExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::MacroInvocation) {}

  void setSimplePath(const SimplePath &p) { path = p; }
  void setPath(std::shared_ptr<ast::Expression> sp) { simplePath = sp; }
  void setTree(std::shared_ptr<DelimTokenTree> tr) { tree = tr; }
};

} // namespace rust_compiler::ast
