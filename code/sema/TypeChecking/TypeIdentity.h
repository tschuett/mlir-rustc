#pragma once

#include "ADT/CanonicalPath.h"
#include "Location.h"

namespace rust_compiler::sema::type_checking {

class TypeIdentity {
  adt::CanonicalPath path;
  Location loc;

public:
  TypeIdentity(const adt::CanonicalPath &path, Location loc)
      : path(path), loc(loc) {}

  static TypeIdentity from(Location loc) {
    return TypeIdentity(adt::CanonicalPath::createEmpty(), loc);
  }

  static TypeIdentity empty() {
    return TypeIdentity(adt::CanonicalPath::createEmpty(),
                        Location::getBuiltinLocation());
  }
};

} // namespace rust_compiler::sema::type_checking
