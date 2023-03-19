#pragma once

#include "Basic/Ids.h"
#include "Location.h"

namespace rust_compiler::tyctx {

class NodeIdentity {
  basic::NodeId nodeId;
  basic::CrateNum crateNum;
  Location loc;

public:
  NodeIdentity(basic::NodeId nodeId, basic::CrateNum crateNum, Location loc)
      : nodeId(nodeId), crateNum(crateNum), loc(loc) {}

  basic::NodeId getNodeId() const { return nodeId; }
  basic::CrateNum getCrate() const { return crateNum; }
  Location getLocation() const { return loc; }
};

} // namespace rust_compiler::tyctx
