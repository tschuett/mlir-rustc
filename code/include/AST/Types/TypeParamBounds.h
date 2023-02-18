#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeParamBound.h"

#include <vector>

namespace rust_compiler::ast::types {

class TypeParamBounds : public Node {

  std::vector<TypeParamBound> typeParamBounds;
  bool trailingPlus;

public:
  TypeParamBounds(Location loc) : Node(loc) {}

  void addTypeParamBound(const TypeParamBound &tpb) {
    typeParamBounds.push_back(tpb);
  }

  bool isTrailingPlus() const { return trailingPlus; }
};

} // namespace rust_compiler::ast::types
