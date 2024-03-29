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

  std::vector<lexer::Token> ts = tree->toTokenStream();

  AttributeParser parser = {ts};

  llvm::errs() << "AttrInput::parseToMetaItem: " << ts.size() << "\n";

  std::vector<std::shared_ptr<MetaItemInner>> items =
      parser.parseMetaItemSequence();

  setMetaItems(std::move(items));
}

// Copy constructor must deep copy MetaItemInner as unique pointer
//AttrInput::AttrInput(const AttrInput &other) : Node(other.getLocation()) {
//  tree = other.tree;
//  expr = other.expr;
//  kind = other.kind;
//  for (unsigned i = 0; i < other.items.size(); ++i)
//    items.push_back(std::shared_ptr<MetaItemInner>(other.items[i]->clone()));
//}

//std::shared_ptr<AttrInput> AttrInput::clone() {
//  AttrInput input = {getLocation()};
//
//  input.tree = tree;
//  input.expr = expr;
//  input.kind = kind;
//  for (unsigned i = 0; i < items.size(); ++i)
//    input.items.push_back(std::shared_ptr<MetaItemInner>(items[i]->clone()));
//
//  return std::shared_unique<AttrInput>(input);
//}

} // namespace rust_compiler::ast
