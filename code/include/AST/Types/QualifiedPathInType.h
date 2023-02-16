#pragma once

#include "AST/Types/QualifiedPathType.h"
#include "AST/Types/TypeNoBounds.h"

#include <vector>

namespace rust_compiler::ast::types {

class QualifiedPathInType : public TypeNoBounds {
  std::vector<QualifiedPathType> segments;

public:
  QualifiedPathInType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::QualifiedPathInType) {}

  void append(const QualifiedPathType &seg) { segments.push_back(seg); }
};

} // namespace rust_compiler::ast::types
