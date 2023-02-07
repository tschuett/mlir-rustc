#pragma once

#include "AST/AST.h"
#include "AST/GenericParam.h"

namespace rust_compiler::ast {

class ConstParam : public GenericParam {
public:
  ConstParam(Location loc) : GenericParam(loc, GenericParamKind::ConstParam) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
