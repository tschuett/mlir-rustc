#pragma once

#include "AST/Types/TypeNoBounds.h"

#include "AST/Types/QualifiedPathType.h"

#include <vector>

namespace rust_compiler::ast::types {

class QualifiedPathInType : public TypeNoBounds {
  std::vector<QualifiedPathType> segmengs;

public:
  QualifiedPathInType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::QualifiedPathInType) {}

  void append(const QualifiedPathType &);
};

} // namespace rust_compiler::ast::types
