#pragma once

#include "AST/AST.h"
#include "AST/SimplePath.h"

#include <optional>
#include <vector>
#include <string>
#include <string_view>

namespace rust_compiler::ast::use_tree {

enum class UseTreeKind {
  Glob,
  Recursive,
  Path,
  Rebinding,
};

class UseTree : public Node {
  std::optional<SimplePath> path;
  std::vector<UseTree> trees;

  bool doubleColon = false;
  bool underscore = false;
  std::string identifier;

  UseTreeKind kind;

public:
  explicit UseTree(Location loc) : Node(loc) {}

  void setPath(const SimplePath &p) { path = p; }
  void setKind(UseTreeKind _kind) { kind = _kind; }
  void setDoubleColon() { doubleColon = true; }
  void addTree(const UseTree &tree) {
    trees.push_back(tree);
    kind = UseTreeKind::Recursive;
  }

  void setIdentifier(std::string_view id) { identifier = id; }
  void setUnderscore() { underscore = true;}
};

} // namespace rust_compiler::ast::use_tree
