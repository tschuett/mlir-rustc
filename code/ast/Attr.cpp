#include "AST/Attr.h"

#include "AST/AttrInput.h"

namespace rust_compiler::ast {

void Attr::parseMetaItem() {
  if (!hasInput())
    return;

  if (isParsedToMetaItem())
    return;

  attrInput->parseToMetaItem();
  parsedToMetaItem = true;
}

bool Attr::isParsedToMetaItem() { return parsedToMetaItem; }

// Copy constructor must deep copy AttrInput as unique pointer
//Attr::Attr(Attr const &other) : Node(other.getLocation()), path(other.path) {
//  attrInput = other.attrInput->clone();
//}

// assignment operator
//Attr &Attr::operator=(const Attr &other) {
//  path = other.path;
//  if (other.attrInput != nullptr)
//    attrInput = other.attrInput->clone();
//  else
//    attrInput = nullptr;
//
//  return *this;
//}

} // namespace rust_compiler::ast
