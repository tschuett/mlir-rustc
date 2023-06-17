#include "AttributeChecker.h"

#include "AST/InnerAttribute.h"
#include "AST/TypeAlias.h"
#include "AST/VisItem.h"
#include "AST/Struct.h"
#include "AST/Trait.h"
#include "AST/Enumeration.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema::attribute_checker {

void AttributeChecker::checkCrate(std::shared_ptr<ast::Crate> crate) {
  std::vector<InnerAttribute> inner = crate->getInnerAttributes();

  for (auto &inn : inner) {
    Attr attr = inn.getAttr();
    attr.parseMetaItem();

    if (attr.hasInput())
      [[maybe_unused]]AttrInput input = attr.getInput();
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
    break;
  }
  case VisItemKind::TypeAlias: {
    checkTypeAlias(static_cast<TypeAlias*>(item));
    break;
  }
  case VisItemKind::Struct: {
    checkStruct(static_cast<Struct*>(item));
    break;
  }
  case VisItemKind::Enumeration: {
    checkEnumeration(static_cast<Enumeration*>(item));
    break;
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
    checkTrait(static_cast<Trait*>(item));
    break;
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
