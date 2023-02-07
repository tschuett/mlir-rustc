#pragma once

#include "AST/AST.h"
#include "AST/GenericParam.h"
#include "AST/Types/TypeParamBounds.h"

#include <optional>
#include <string>

namespace rust_compiler::ast {

class TypeParam : public GenericParam {
  std::string identifier;
  std::optional<types::TypeParamBounds> bounds;
  std::optional<std::shared_ptr<types::Type>> type;

public:
  TypeParam(Location loc) : GenericParam(loc, GenericParamKind::TypeParam) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
