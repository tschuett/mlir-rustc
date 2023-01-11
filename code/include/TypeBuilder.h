#pragma once

#include "AST/Types/Types.h"

#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace rust_compiler {

class TypeBuilder {
  mlir::Builder builder;

public:
  TypeBuilder(mlir::Builder builder) : builder(builder) {}
  mlir::Type getType(std::shared_ptr<ast::types::Type>);
};

} // namespace rust_compiler
