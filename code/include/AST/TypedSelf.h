#pragma once

#include "AST/AST.h"
#include "AST/SelfParam.h"
#include "AST/Type.h"

namespace rust_compiler::ast {

class TypedSelf : public SelfParam {
  bool mut;
  std::shared_ptr<ast::types::Type> type;

public:
  TypedSelf(Location loc) : SelfParam(loc) {}

  void setMut();
  void setType(std::shared_ptr<ast::types::Type> type);

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
