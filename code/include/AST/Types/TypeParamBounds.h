#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeParamBound.h"

#include <vector>
#include <memory>

namespace rust_compiler::ast::types {

class TypeParamBounds : public Node {

  std::vector<std::shared_ptr<TypeParamBound>> typeParamBounds;
  bool trailingPlus;

public:
  TypeParamBounds(Location loc) : Node(loc) {}

  void addTypeParamBound(const std::shared_ptr<TypeParamBound> &tpb) {
    typeParamBounds.push_back(tpb);
  }

  bool isTrailingPlus() const { return trailingPlus; }
};

} // namespace rust_compiler::ast::types
