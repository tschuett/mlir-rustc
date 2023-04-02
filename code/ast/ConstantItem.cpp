#include "AST/ConstantItem.h"

namespace rust_compiler::ast {

void ConstantItem::setIdentifier(const Identifier& id) { identifier = id; }

void ConstantItem::setType(std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}

void ConstantItem::setInit(std::shared_ptr<Expression> _init) { init = _init; }

} // namespace rust_compiler::ast
