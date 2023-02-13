#pragma once

#include "AST/AST.h"
#include "AST/SimplePath.h"

namespace rust_compiler::ast::use_tree {

class UseTree : public Node {
public:
  explicit UseTree(Location loc) : Node(loc) {}
};

/// SimplePath
class SimplePathNode : public UseTree {
  SimplePath path;

public:
  explicit SimplePathNode(Location loc) : UseTree(loc), path(loc) {}

  void setSimplePath(SimplePath path);
};

/// { ... }
class PathList : public UseTree {
  std::vector<std::shared_ptr<UseTree>> elements;

public:
  explicit PathList(Location loc) : UseTree(loc) {}


  void addTree(std::shared_ptr<UseTree> tree);
};

class Star : public UseTree {
public:
  explicit Star(Location loc) : UseTree(loc) {}

};

/// :: *
class DoubleColonStar : public UseTree {
public:
  explicit DoubleColonStar(Location loc) : UseTree(loc) {}

};

/// SimplePath :: *
class SimplePathDoubleColonStar : public UseTree {
public:
  explicit SimplePathDoubleColonStar(Location loc) : UseTree(loc) {}

  void setPath(SimplePath path);
};

/// SimplePath :: { ... };
class SimplePathDoubleColonWithPathList : public UseTree {
  PathList list;

public:
  explicit SimplePathDoubleColonWithPathList(Location loc) : UseTree(loc), list(loc) {}

  void setPathList(PathList list);
};

/// :: { ... }
class DoubleColonWithPathList : public UseTree {
  PathList list;

public:
  explicit DoubleColonWithPathList(Location loc) : UseTree(loc), list(loc) {}

  // void append(SimplePath path);
};

/// foo as bar
class Rebinding : public UseTree {
public:
  explicit Rebinding(Location loc) : UseTree(loc) {}

};

} // namespace rust_compiler::ast::use_tree
