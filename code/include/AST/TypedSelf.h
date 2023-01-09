#pragma once

#include "AST/AST.h"
#include "AST/SelfParam.h"
#include "AST/Type.h"
#include "AST/Types/Types.h"

namespace rust_compiler::ast {

class TypedSelf : public SelfParam {
  bool mut = false;
  std::shared_ptr<ast::types::Type> type;

public:
  TypedSelf(Location loc) : SelfParam(loc) {}

  void setMut() { mut = true; }
  void setType(std::shared_ptr<ast::types::Type> _type) { type = _type; }

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
