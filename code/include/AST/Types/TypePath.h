#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePathSegment.h"
#include "AST/Types/Types.h"

#include <vector>

namespace rust_compiler::ast::types {

class TypePath final : public TypeNoBounds {
  bool leadingDoubleColon = false;
  std::vector<TypePathSegment> typePathSegments;

public:
  TypePath(Location loc) : TypeNoBounds(loc, TypeNoBoundsKind::TypePath) {}

  void setLeadingPathSep() { leadingDoubleColon = true; }

  void addSegment(const TypePathSegment &seg) {
    typePathSegments.push_back(seg);
  }
};

} // namespace rust_compiler::ast::types
