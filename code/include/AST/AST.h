#pragma once

#include "Basic/Ids.h"
#include "Location.h"
#include "TyCtx/TyCtx.h"

#include <cstddef>
#include <vector>

// https://doc.rust-lang.org/nightly/nightly-rustc/rustc_ast/ast/index.html

namespace rust_compiler::ast {

/// Base class for all AST nodes except the Crate. Each node has an id and a location.
class Node {
  Location location;
  basic::NodeId nodeId;

public:
  explicit Node(Location location) : location(location) {
    nodeId = tyctx::TyCtx::get()->getNextNodeId();
  }

  virtual ~Node() = default;

  rust_compiler::Location getLocation() const { return location; }
  basic::NodeId getNodeId() const{ return nodeId;}
};

} // namespace rust_compiler::ast
