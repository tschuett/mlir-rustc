#pragma once

#include "AST/AST.h"
#include "AST/GenericParam.h"

namespace rust_compiler::ast {

class LifetimeParam : public GenericParam {
public:
  LifetimeParam(Location loc)
      : GenericParam(loc, GenericParamKind::LifetimeParam) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
