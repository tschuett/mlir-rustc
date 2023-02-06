#include "ADT/ScopedCanonicalPath.h"

#include <algorithm>

namespace rust_compiler::adt {

ScopedCanonicalPathScope::ScopedCanonicalPathScope(ScopedCanonicalPath *storage,
                                                   std::string_view segment) {
  storage->registerScope(this, segment);
  parent = storage;
};

ScopedCanonicalPathScope::~ScopedCanonicalPathScope() {
  parent->deregisterScope(this);
}

CanonicalPath ScopedCanonicalPath::getCurrentPath() const {
  CanonicalPath canPath = {crateName};

  std::stack<std::string> copy(segments);
  std::vector<std::string> result;

  for (unsigned i = 0; i < copy.size(); ++i) {
    result.push_back(copy.top());
    copy.pop();
  }

  std::reverse(result.begin(), result.end());

  canPath.segments = result;

  return canPath;
}

void ScopedCanonicalPath::registerScope(ScopedCanonicalPathScope *,
                                        std::string_view segment) {
  assert(false);
}
void ScopedCanonicalPath::deregisterScope(ScopedCanonicalPathScope *) {
  assert(false);
}

} // namespace rust_compiler::adt
