#include "AST/Attr.h"

#include "AST/AttrInput.h"

namespace rust_compiler::ast {

void Attr::parseMetaItem() {
  if (!hasAttrInput())
    return;

  if (isParsedToMetaItem())
    return;

  attrInput->parseToMetaItem();
}

} // namespace rust_compiler::ast
