#pragma once

#include "ADT/CanonicalPath.h"
#include "AST/InnerAttribute.h"
#include "AST/Item.h"
#include "AST/Module.h"
#include "Basic/Ids.h"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace rust_compiler::ast {

/// A crate AST object - holds all the data for a single compilation unit
class Crate {
  std::vector<InnerAttribute> innerAttributes;

  std::vector<std::shared_ptr<Item>> items;

  std::string crateName;

  basic::CrateNum crateNum;

  basic::NodeId nodeId;

public:
  Crate(std::string_view crateName, basic::CrateNum crateNum);

  /// how?
  void merge(std::shared_ptr<ast::Module> module, adt::CanonicalPath path);

  std::vector<std::shared_ptr<Item>> getItems() const { return items; }

  std::string getCrateName() const;

  basic::CrateNum getCrateNum() const { return crateNum; }

  basic::NodeId getNodeId() const { return nodeId; }

  void setInnerAttributes(std::span<InnerAttribute> attr) {
    innerAttributes = {attr.begin(), attr.end()};
  }

  void addItem(std::shared_ptr<Item> it) { items.push_back(it); }

  std::vector<InnerAttribute> getInnerAttributes() const {
    return innerAttributes;
  }

  std::optional<basic::NodeId> getOwnerItem(basic::NodeId, ast::Item *);
};

} // namespace rust_compiler::ast
