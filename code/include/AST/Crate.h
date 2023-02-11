#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/InnerAttribute.h"
#include "AST/Item.h"
#include "AST/Module.h"
#include "Basic/Ids.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

/// A crate AST object - holds all the data for a single compilation unit
class Crate {
  std::vector<std::shared_ptr<InnerAttribute>> inner_attrs;
  // dodgy spacing required here
  /* TODO: is it better to have a vector of items here or a module (implicit
   * top-level one)? */
  std::vector<std::shared_ptr<Item>> items;

  std::string crateName;

  basic::CrateNum crateNum;

public:
  Crate(std::string_view crateName, basic::CrateNum crateNum)
      : crateName(crateName), crateNum(crateNum){};

  void merge(std::shared_ptr<ast::Module> module, adt::CanonicalPath path);

  std::span<std::shared_ptr<Item>> getItems() const;

  std::string_view getCrateName() const;
};

} // namespace rust_compiler::ast
