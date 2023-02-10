#pragma once

#include "Basic/Ids.h"

#include <stack>
#include <string_view>
#include <vector>

namespace rust_compiler::sema::resolver {

class Rib {
public:
  Rib(std::string_view crateName, basic::NodeId node_id);
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
  ~Resolver() = default;

  void pushNewModuleScope(basic::NodeId);
  void popNewModuleScope(basic::NodeId);
  basic::NodeId peekCurrentModuleScope();
private:
  Resolver();

  Scope nameScope;
  Scope typeScope;
  Scope labelScope;
  Scope macroScope;

  // keep track of the current module scope ids
  std::stack<basic::NodeId> currentModuleStack;
};

} // namespace rust_compiler::sema::resolver
