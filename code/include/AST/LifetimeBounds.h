#pragma once

#include "AST/AST.h"
#include "AST/Lifetime.h"

namespace rust_compiler::ast {

class LifetimeBounds : public Node {
  bool trailingPlus = false;
  std::vector<Lifetime> lifetimes;

public:
  LifetimeBounds(Location loc) : Node(loc) {}

  bool isTrailingPlus() const { return trailingPlus; }

  void setLifetime(const Lifetime &lf) { lifetimes.push_back(lf); }

  void setTrailingPlus() { trailingPlus = true; }
};

} // namespace rust_compiler::ast
