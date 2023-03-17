#pragma once

#include "AST/AST.h"
#include "AST/SimplePathSegment.h"

#include <cstddef>
#include <llvm/ADT/SmallVector.h>

namespace rust_compiler::ast {

class SimplePath : public Node {
  llvm::SmallVector<SimplePathSegment> segments;
  bool withDoubleColon = false;

public:
  explicit SimplePath(Location loc) : Node(loc) {}
  void setWithDoubleColon();
  void addPathSegment(SimplePathSegment &seg);

  size_t getNrOfSegments() const { return segments.size(); }
  SimplePathSegment getSegment(size_t i) const { return segments[i]; }
};

} // namespace rust_compiler::ast
