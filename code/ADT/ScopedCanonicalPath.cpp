#include "ADT/ScopedCanonicalPath.h"

#include <algorithm>

namespace rust_compiler::adt {

ScopedCanonicalPathScope::ScopedCanonicalPathScope(ScopedCanonicalPath *storage,
                                                   basic::NodeId nodeId,
                                                   std::string_view segment) {
  storage->registerScope(this, nodeId, segment);
  parent = storage;
};

ScopedCanonicalPathScope::~ScopedCanonicalPathScope() {
  parent->deregisterScope(this);
}

CanonicalPath ScopedCanonicalPath::getCurrentPath() const {}

void ScopedCanonicalPath::registerScope(ScopedCanonicalPathScope *scope,
                                        basic::NodeId nodeId,
                                        std::string_view segment) {
  std::pair<basic::NodeId, std::string> p = std::make_pair(nodeId, std::string(segment));
  scopes.push(scope);
  segments.push(p);
  assert(false);
}

void ScopedCanonicalPath::deregisterScope(ScopedCanonicalPathScope *scope) {
  scopes.pop();
  segments.pop();
  assert(false);
}

} // namespace rust_compiler::adt
