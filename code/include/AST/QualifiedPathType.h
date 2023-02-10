#pragma once

#include "AST/Types/Types.h"
#include "Location.h"

namespace rust_compiler::ast::types {

class QualifiedPathType final : public Type {
public:
  size_t getTokens() override;
};

} // namespace rust_compiler::ast::types
