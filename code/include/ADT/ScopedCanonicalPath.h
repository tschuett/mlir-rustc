#pragma once

#include "ADT/CanonicalPath.h"

#include <stack>
#include <string>
#include <string_view>

namespace rust_compiler::adt {

class ScopedCanonicalPath;

class ScopedCanonicalPathScope {
  ScopedCanonicalPath *parent;

public:
  ScopedCanonicalPathScope(ScopedCanonicalPath *storage, basic::NodeId id,
                           std::string_view segment);

  ~ScopedCanonicalPathScope();
};

class ScopedCanonicalPath {
  using ScopeTy = ScopedCanonicalPathScope;

  std::stack<ScopeTy *> scopes;

  std::stack<std::pair<basic::NodeId, std::string>> segments;
  std::string crateName;

  CanonicalPath path;

public:
  ScopedCanonicalPath(const CanonicalPath &path) : path(path) {
    crateName = path.getCrateName();
  }

  CanonicalPath getCurrentPath() const;

private:
  friend ScopedCanonicalPathScope;

  void registerScope(ScopedCanonicalPathScope *, basic::NodeId nodeId,
                     std::string_view segment);
  void deregisterScope(ScopedCanonicalPathScope *);
};

} // namespace rust_compiler::adt
