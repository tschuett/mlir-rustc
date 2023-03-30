#include "AST/AttrInput.h"

#include "AST/AttributeParser.h"
#include "AST/DelimTokenTree.h"
#include "Lexer/TokenStream.h"

namespace rust_compiler::ast {

void AttrInput::parseToMetaItem() {
  if (kind == AttrInputKind::Expression)
    return;

  if (tree->isEmpty())
    return;

  lexer::TokenStream ts = tree->toTokenStream();

  AttributeParser parser = {ts};

  std::vector<std::unique_ptr<MetaItemInner>> items =
      parser.parseMetaItemSequence();

  setMetaItems(std::move(items));
}

} // namespace rust_compiler::ast
