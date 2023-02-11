#pragma once

#include "Basic/Ids.h"

#include <map>
#include <stack>
#include <string_view>
#include <vector>

namespace rust_compiler::sema::resolver {

class Rib {
public:
  Rib(basic::CrateNum crateNum, basic::NodeId nodeId)
      : crateNum(crateNum), nodeId(nodeId) {}

  basic::NodeId getNodeId() const { return nodeId; }

private:
  basic::CrateNum crateNum;
  basic::NodeId nodeId;
};

class Scope {
public:
  Scope(basic::CrateNum crateNum);

  Rib *peek();
  void push(basic::NodeId id);

  basic::CrateNum getCrateNum() const { return crateNum; }

private:
  basic::CrateNum crateNum;
  basic::NodeId node_id;
  std::vector<Rib *> stack;
};

class Resolver {
public:
  static Resolver *get();
  ~Resolver() = default;

  // these builtin types
  void insertBuiltinTypes(Rib *r);

  void pushNewTypeRib(Rib *);

  void pushNewModuleScope(basic::NodeId);
  void popNewModuleScope(basic::NodeId);
  basic::NodeId peekCurrentModuleScope();

  Scope &getTypeScope() { return typeScope; }

private:
  Resolver();

  /// ?
  basic::NodeId globalTypeNodeId;

  Scope nameScope;
  Scope typeScope;
  Scope labelScope;
  Scope macroScope;

  // map a AST Node to a Rib
  std::map<basic::NodeId, Rib *> nameRibs;
  std::map<basic::NodeId, Rib *> typeRibs;
  std::map<basic::NodeId, Rib *> labelRibs;
  std::map<basic::NodeId, Rib *> macroRibs;

  // keep track of the current module scope ids
  std::stack<basic::NodeId> currentModuleStack;
};

} // namespace rust_compiler::sema::resolver
