#pragma once

#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePathSegment.h"

#include <vector>

namespace rust_compiler::ast::types {

class TypePath final : public TypeNoBounds {
  bool leadingDoubleColon = false;
  std::vector<TypePathSegment> typePathSegments;

public:
  TypePath(Location loc) : TypeNoBounds(loc, TypeNoBoundsKind::TypePath) {}

  void setLeadingPathSep() { leadingDoubleColon = true; }

  bool hasLeadingPathSep() const { return leadingDoubleColon; }
  void addSegment(const TypePathSegment &seg) {
    typePathSegments.push_back(seg);
  }

  std::vector<TypePathSegment> getSegments() const { return typePathSegments; }
};

} // namespace rust_compiler::ast::types
