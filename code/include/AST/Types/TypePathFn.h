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
  std::optional<std::shared_ptr<types::TypeExpression>> type;

public:
  TypePathFn(Location loc) : Node(loc) {}

  void setTypePathFnInputs(ast::types::TypePathFnInputs in) {
    typePathFnInputs = in;
  };

  void setType(std::shared_ptr<types::TypeExpression> ex) { type = ex; }

  bool hasInputs() const { return typePathFnInputs.has_value(); }
  bool hasType() const { return type.has_value(); }

  std::shared_ptr<types::TypeExpression> getType() const { return *type; };
  TypePathFnInputs getInputs() const { return *typePathFnInputs; }
};

} // namespace rust_compiler::ast::types
