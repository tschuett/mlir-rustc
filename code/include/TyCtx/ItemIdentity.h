#pragma once

#include "ADT/CanonicalPath.h"
#include "Location.h"

namespace rust_compiler::tyctx {

class ItemIdentity {
  adt::CanonicalPath path;
  Location loc;

public:
  ItemIdentity(adt::CanonicalPath &path, Location loc) : path(path), loc(loc) {}

  adt::CanonicalPath getPath() const { return path; }
};

} // namespace rust_compiler::tyctx
