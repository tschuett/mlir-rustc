#include "Basic/Ids.h"

namespace rust_compiler::basic {

NodeId getNextNodeId() {
  static NodeId iter = 7;
  ++iter;
  return iter;
}

} // namespace rust_compiler::basic
