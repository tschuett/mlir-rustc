#include "AST/AttrInput.h"

#include "AST/AttributeParser.h"
#include "AST/DelimTokenTree.h"
#include "Lexer/TokenStream.h"

#include <memory>

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

// Copy constructor must deep copy MetaItemInner as unique pointer
AttrInput::AttrInput(const AttrInput &other) : Node(other.getLocation()) {
  tree = other.tree;
  expr = other.expr;
  kind = other.kind;
  for (unsigned i = 0; i < other.items.size(); ++i)
    items.push_back(std::unique_ptr<MetaItemInner>(other.items[i]->clone()));
}

std::unique_ptr<AttrInput> AttrInput::clone() {
  AttrInput input = {getLocation()};

  input.tree = tree;
  input.expr = expr;
  input.kind = kind;
  for (unsigned i = 0; i < items.size(); ++i)
    input.items.push_back(std::unique_ptr<MetaItemInner>(items[i]->clone()));

  return std::make_unique<AttrInput>(input);
}

} // namespace rust_compiler::ast
