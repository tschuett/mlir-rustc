#pragma once

#include "Basic/Ids.h"
#include "Location.h"
#include "TyCtx/NodeIdentity.h"
#include "TyCtx/TyCtx.h"

#include <cstddef>
#include <vector>

// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/index.html

namespace rust_compiler::ast {

/// Base class for all AST nodes except the Crate. Each node has an id and a
/// location.
class Node {
  Location location;
  basic::NodeId nodeId;
  basic::CrateNum crateNum;

public:
  explicit Node(Location location) : location(location) {
    nodeId = rust_compiler::tyctx::TyCtx::get()->getNextNodeId();
    crateNum = rust_compiler::tyctx::TyCtx::get()->getCurrentCrate();
  }

  virtual ~Node() = default;

  rust_compiler::Location getLocation() const { return location; }
  basic::NodeId getNodeId() const { return nodeId; }

  tyctx::NodeIdentity getIdentity() const {
    return tyctx::NodeIdentity(nodeId, crateNum, location);
  }
};

} // namespace rust_compiler::ast
