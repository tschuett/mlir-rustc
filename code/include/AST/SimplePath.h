#pragma once

#include "AST/AST.h"
#include "AST/SimplePathSegment.h"

#include <cstddef>
#include <vector>

namespace rust_compiler::ast {

class SimplePath : public Node {
  std::vector<SimplePathSegment> segments;
  bool withDoubleColon = false;

public:
  void setWithDoubleColon();
  void addPathSegment(SimplePathSegment &seg);

  size_t getTokens() override;

  std::string toString();
};

} // namespace rust_compiler::ast
