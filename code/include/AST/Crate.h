#pragma once

#include "AST/CanonicalPath.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

class InnerAttribute;
class Item;
class Module;

// A crate AST object - holds all the data for a single compilation unit
class Crate {
  std::vector<InnerAttribute> inner_attrs;
  // dodgy spacing required here
  /* TODO: is it better to have a vector of items here or a module (implicit
   * top-level one)? */
  std::vector<std::unique_ptr<Item>> items;

  std::string crateName;

public:
  Crate(std::string_view crateName) : crateName(crateName){};

  void merge(std::shared_ptr<ast::Module> module, CanonicalPath path);

  std::span<std::unique_ptr<Item>> getItems() const;
};

} // namespace rust_compiler::ast