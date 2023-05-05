#pragma once

#include "AST/AssociatedItem.h"
#include "Basic/Ids.h"

namespace rust_compiler::ast {
class Trait;
} // namespace rust_compiler::ast

namespace rust_compiler::tyctx {

/// a reference to a Trait
class TraitReference {

public:
  ast::Trait *getTrait() const { return trait; }

private:
  ast::Trait *trait;
};

class AssociatedItemReference {
public:
  ast::AssociatedItem *getItem() const { return item; }
  basic::NodeId getNodeId() const { return item->getNodeId(); }

private:
  ast::AssociatedItem *item;
};

} // namespace rust_compiler::tyctx
