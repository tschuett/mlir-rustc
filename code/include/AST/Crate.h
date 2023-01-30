#pragma once

#include <memory>
#include <span>
#include <vector>

namespace rust_compiler::ast {

class Attribute;
class Item;

// A crate AST object - holds all the data for a single compilation unit
class Crate {
  std::vector<Attribute> inner_attrs;
  // dodgy spacing required here
  /* TODO: is it better to have a vector of items here or a module (implicit
   * top-level one)? */
  std::vector<std::unique_ptr<Item>> items;

public:
  std::span<std::unique_ptr<Item>> getItems() const;
};

} // namespace rust_compiler::ast
