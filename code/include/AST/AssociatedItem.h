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

enum class AssociatedItemKind {
  MacroInvocationSemi,
  TypeAlias,
  ConstantItem,
  Function
};

class AssociatedItem : public Node {
  std::vector<OuterAttribute> outerAttributes;
  std::shared_ptr<ast::Item> macroItem;
  std::shared_ptr<ast::Item> typeAlias;
  std::shared_ptr<ast::Item> constantItem;
  std::shared_ptr<ast::Item> function;
  std::optional<Visibility> visibility;

  AssociatedItemKind kind;

public:
  AssociatedItem(Location loc) : Node(loc) {}

  void setOuterAttributes(std::span<OuterAttribute> o) {
    outerAttributes = {o.begin(), o.end()};
  }
  void setConstantItem(std::shared_ptr<ast::Item> co) {
    kind = AssociatedItemKind::ConstantItem;
    constantItem = co;
  }
  void setTypeAlias(std::shared_ptr<ast::Item> type) {
    kind = AssociatedItemKind::TypeAlias;
    typeAlias = type;
  }
  void setFunction(std::shared_ptr<ast::Item> f) {
    kind = AssociatedItemKind::Function;
    function = f;
  }
  void setVisiblity(Visibility vi) { visibility = vi; }
  void setMacroItem(std::shared_ptr<ast::Item> m) {
    kind = AssociatedItemKind::MacroInvocationSemi;
    macroItem = m;
  }

  AssociatedItemKind getKind() const { return kind; }

  std::optional<Visibility> getVisibility() const { return visibility; }

  bool hasFunction() const { return (bool)function; }
  bool hasTypeAlias() const { return (bool)typeAlias; }
  bool hasConstantItem() const { return (bool)constantItem; }
  bool hasMacroInvocationSemi() const { return (bool)macroItem; }

  std::shared_ptr<ast::Item> getTypeAlias() const { return typeAlias; }
  std::shared_ptr<ast::Item> getConstantItem() const { return constantItem; }
  std::shared_ptr<ast::Item> getFunction() const { return function; }
  std::shared_ptr<ast::Item> getMacroItem() const { return macroItem; }
};

} // namespace rust_compiler::ast
