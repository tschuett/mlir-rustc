#include "AST/AST.h"

#include "TyCtx/TyCtx.h"

namespace rust_compiler::ast {

Node::Node(Location location) : location(location) {
  nodeId = rust_compiler::tyctx::TyCtx::get()->getNextNodeId();
  crateNum = rust_compiler::tyctx::TyCtx::get()->getCurrentCrate();
}

} // namespace rust_compiler::ast
