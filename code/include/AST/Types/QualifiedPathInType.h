#pragma once

#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePathSegment.h"

#include <vector>

namespace rust_compiler::ast::types {

class QualifiedPathInType : public TypeNoBounds {
  QualifiedPathType first;
  std::vector<TypePathSegment> segments;

public:
  QualifiedPathInType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::QualifiedPathInType), first(loc) {}

  void setSegment(const QualifiedPathType &pat) { first = pat; }
  void append(const TypePathSegment &seg) { segments.push_back(seg); }
};

} // namespace rust_compiler::ast::types
