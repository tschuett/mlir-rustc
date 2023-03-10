#pragma once

#include "AST/AST.h"
#include "AST/ConstantItem.h"
#include "AST/Function.h"
#include "AST/MacroInvocationSemiItem.h"
#include "AST/MacroItem.h"
#include "AST/OuterAttribute.h"
#include "AST/TypeAlias.h"

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class AssociatedItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<ast::Item> macroItem;
  std::shared_ptr<ast::Item> typeAlias;
  std::shared_ptr<ast::Item> constantItem;
  std::shared_ptr<ast::Item> function;
  std::optional<Visibility> visibility;

public:
  AssociatedItem(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }
  void setConstantItem(std::shared_ptr<ast::Item> co) { constantItem = co; }
  void setTypeAlias(std::shared_ptr<ast::Item> type) { type = typeAlias; }
  void setFunction(std::shared_ptr<ast::Item> f) { function = f; }
  void setVisiblity(Visibility vi) { visibility = vi; }
  void setMacroItem(std::shared_ptr<ast::Item> m) { macroItem = m; }
};

} // namespace rust_compiler::ast
