#include "ADT/CanonicalPath.h"

namespace rust_compiler::adt {

bool CanonicalPath::isEqualByName(const CanonicalPath &b) const {
  if (segments.size() != b.segments.size())
    return false;

  for (unsigned i = 0; i < segments.size(); ++i)
    if (segments[i].second != b.segments[i].second)
      return false;

  return true;
}

} // namespace rust_compiler::adt
