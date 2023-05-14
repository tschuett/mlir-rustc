#pragma once

#include "Basic/Ids.h"
#include "Location.h"
#include "TyCtx/NodeIdentity.h"

// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/index.html

namespace rust_compiler::ast {

/// Base class for all AST nodes except the Crate. Each node has an id and a
/// location.
class Node {
  Location location;
  basic::NodeId nodeId;
  basic::CrateNum crateNum;

  unsigned constant : 1;
  unsigned place : 1;

public:
  explicit Node(Location location);

  virtual ~Node() = default;

  rust_compiler::Location getLocation() const { return location; }
  basic::NodeId getNodeId() const { return nodeId; }

  tyctx::NodeIdentity getIdentity() const {
    return tyctx::NodeIdentity(nodeId, crateNum, location);
  }

  void setConstant() { constant = 1; }
  bool isConstant() const { return constant; }

  void setPlaceExpression() { place = 1; }
  bool getPlaceExpression() { return place; }
};

} // namespace rust_compiler::ast
