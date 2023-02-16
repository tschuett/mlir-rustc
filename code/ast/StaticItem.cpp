#include "AST/StaticItem.h"

#include <llvm/Support/raw_ostream.h>

namespace rust_compiler::ast {

void StaticItem::setMut() { mut = true; }

void StaticItem::setIdentifier(std::string_view _id) { identifier = _id; }

void StaticItem::setType(std::shared_ptr<types::TypeExpression> _type) {
  type = _type;
}
void StaticItem::setInit(std::shared_ptr<Expression> _init) { init = _init; }

} // namespace rust_compiler::ast
