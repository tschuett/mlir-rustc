#include "AST/ConstantItem.h"

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::ast {

void ConstantItem::setIdentifier(std::string_view id) { identifier = id; }

void ConstantItem::setType(std::shared_ptr<ast::types::TypeExpression> _type) {
  type = _type;
}

void ConstantItem::setInit(std::shared_ptr<Expression> _init) { init = _init; }

} // namespace rust_compiler::ast
