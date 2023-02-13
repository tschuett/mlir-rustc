#pragma once

#include "AST/AST.h"
#include "AST/Types/Types.h"

#include <memory>
#include <string>

namespace rust_compiler::ast {

class GenericArgsBinding : public Node {
  std::string identifier;
  std::shared_ptr<types::Type> type;

public:
};

} // namespace rust_compiler::ast
