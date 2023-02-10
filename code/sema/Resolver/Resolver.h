#pragma once

#include "Sema/Common.h"

#include <stack>
#include <string_view>
#include <vector>

namespace rust_compiler::sema::resolver {
class Rib {
public:
  Rib(std::string_view crateName, NodeId node_id);
};

class Scope {
public:
  Scope(std::string_view crateName);

private:
  std::vector<Rib *> ribs;
};

class Resolver {
public:
  static Resolver *get();
  ~Resolver() {}

private:
  Resolver();

  Scope nameScope;
  Scope typeScope;
  Scope labelScope;
  Scope macroScope;
};

} // namespace rust_compiler::sema::resolver
