#pragma once

#include "AST/AST.h"
#include "AST/GenericArgs.h"
#include "AST/Types/TypePathFn.h"
#include "AST/Types/TypePathFnInputs.h"
#include "AST/Types/Types.h"

#include <optional>
#include <vector>

namespace rust_compiler::ast::types {

class TypePathFn final : public Node {
  std::optional<TypePathFnInputs> typePathFnInputs;
  std::optional<std::shared_ptr<types::Type>> type;

public:
  TypePathFn(Location loc) : Node(loc) {}
};

} // namespace rust_compiler::ast::types
