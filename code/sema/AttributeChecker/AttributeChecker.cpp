#include "AttributeChecker.h"

#include "AST/InnerAttribute.h"
#include "AST/VisItem.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema::attribute_checker {

void AttributeChecker::checkCrate(std::shared_ptr<ast::Crate> crate) {
  assert(false);

  std::vector<InnerAttribute> inner = crate->getInnerAttributes();

  for (auto &inn : inner) {
    Attr &attr = inn.getAttr();
    attr.parseMetaItem();

    AttrInput *input = attr.getInput();
  }

  std::vector<std::shared_ptr<Item>> items = crate->getItems();
  for (auto item : items) {
    checkItem(item.get());
  }
}

void AttributeChecker::checkItem(Item *item) {
  switch (item->getItemKind()) {
  case ItemKind::VisItem: {
    checkVisItem(static_cast<VisItem *>(item));
    break;
  }
  case ItemKind::MacroItem: {
    assert(false);
  }
  }
}

void AttributeChecker::checkVisItem(VisItem *item) {
  switch (item->getKind()) {
  case VisItemKind::Module: {
    assert(false);
  }
  case VisItemKind::ExternCrate: {
    assert(false);
  }
  case VisItemKind::UseDeclaration: {
    assert(false);
  }
  case VisItemKind::Function: {
    checkFunction(static_cast<Function*>(item));
    assert(false);
  }
  case VisItemKind::TypeAlias: {
    assert(false);
  }
  case VisItemKind::Struct: {
    assert(false);
  }
  case VisItemKind::Enumeration: {
    assert(false);
  }
  case VisItemKind::Union: {
    assert(false);
  }
  case VisItemKind::ConstantItem: {
    assert(false);
  }
  case VisItemKind::StaticItem: {
    assert(false);
  }
  case VisItemKind::Trait: {
    assert(false);
  }
  case VisItemKind::Implementation: {
    assert(false);
  }
  case VisItemKind::ExternBlock: {
    assert(false);
  }
  }
}

} // namespace rust_compiler::sema::attribute_checker
