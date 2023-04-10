#include "AST/AST.h"

#include "Basic/Ids.h"
#include "Session/Session.h"

using namespace rust_compiler::session;

namespace rust_compiler::ast {

Node::Node(Location location) : location(location) {
  nodeId = rust_compiler::basic::getNextNodeId();
  crateNum = rust_compiler::session::session->getCurrentCrateNum();
}

} // namespace rust_compiler::ast
