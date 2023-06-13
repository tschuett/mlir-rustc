#include "AST/OuterAttribute.h"

namespace rust_compiler::ast {

void OuterAttribute::parseToMetaItem() { attr.parseMetaItem(); }

Attr &OuterAttribute::getAttr() { return attr; }

} // namespace rust_compiler::ast
