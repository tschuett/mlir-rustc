#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/InnerAttribute.h"
#include "AST/Item.h"
#include "AST/Module.h"
#include "Basic/Ids.h"
#include "Mappings/Mappings.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

/// A crate AST object - holds all the data for a single compilation unit
class Crate {
  std::vector<InnerAttribute> innerAttributes;
  // dodgy spacing required here
  /* TODO: is it better to have a vector of items here or a module (implicit
   * top-level one)? */
  std::vector<std::shared_ptr<Item>> items;

  std::string crateName;

  basic::CrateNum crateNum;

  basic::NodeId nodeId;

public:
  Crate(std::string_view crateName, basic::CrateNum crateNum)
      : crateName(crateName), crateNum(crateNum) {
    nodeId = mappings::Mappings::get()->getNextNodeId();
  };

  /// how?
  void merge(std::shared_ptr<ast::Module> module, adt::CanonicalPath path);

  std::vector<std::shared_ptr<Item>> getItems() { return items; }

  std::string_view getCrateName() const;

  basic::CrateNum getCrateNum() const { return crateNum; }

  basic::NodeId getNodeId() const { return nodeId; }

  void setInnerAttributes(std::span<InnerAttribute> attr) {
    innerAttributes = {attr.begin(), attr.end()};
  }

  void addItem(std::shared_ptr<Item> it) { items.push_back(it); }
};

} // namespace rust_compiler::ast
