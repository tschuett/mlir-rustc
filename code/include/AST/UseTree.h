#pragma once

#include "AST/AST.h"
#include "AST/SimplePath.h"

namespace rust_compiler::ast::use_tree {

class UseTree : public Node {
public:
};

/// SimplePath
class SimplePathNode : public UseTree {
public:
  size_t getTokens() override;
};

/// { ... }
class PathList : public UseTree {
  std::vector<std::shared_ptr<UseTree>> elements;

public:
  size_t getTokens() override;

  void addTree(std::shared_ptr<UseTree> tree);
};

class Star : public UseTree {
public:
  size_t getTokens() override;
};

/// :: *
class DoubleColonStar : public UseTree {
public:
  size_t getTokens() override;
};

/// SimplePath :: *
class SimplePathDoubleColonStar : public UseTree {
public:
  size_t getTokens() override;

  void setPath(SimplePath path);
};

/// SimplePath :: { ... }
class SimplePathDoubleColonWithPathList : public UseTree {
  PathList list;

public:
  SimplePathDoubleColonWithPathList() = default;
  size_t getTokens() override;

  void setPathList(PathList list);
};

/// :: { ... }
class DoubleColonWithPathList : public UseTree {
  PathList list;

public:
  size_t getTokens() override;

  // void append(SimplePath path);
};

/// foo as bar
class Rebinding : public UseTree {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast::use_tree
