#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/Expression.h"
#include "AST/SimplePath.h"

namespace rust_compiler::ast {

class MacroInvocationExpression : public ExpressionWithoutBlock {
  SimplePath simplePath;
  std::shared_ptr<DelimTokenTree> tree;

public:
  MacroInvocationExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::MacroInvocation),
    simplePath(loc) {}

  void setPath(const SimplePath &sp) { simplePath = sp; }
  void setTree(std::shared_ptr<DelimTokenTree> tr) { tree = tr; }
};

} // namespace rust_compiler::ast
