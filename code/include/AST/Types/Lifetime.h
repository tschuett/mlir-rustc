#pragma once

#include "AST/AST.h"
#include "AST/Types/TypeParamBound.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast::types {

class Lifetime : public TypeParamBound {
  std::string lifeTime;

public:
  Lifetime(Location loc) : TypeParamBound(TypeParamBoundKind::Lifetime, loc) {}

  void setLifetime(std::string_view s) { lifeTime = s; }
};

} // namespace rust_compiler::ast::types
