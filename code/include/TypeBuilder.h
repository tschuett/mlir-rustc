#pragma once

#include "AST/Types/Types.h"
#include "ModuleBuilder/Target.h"

#include <memory>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

namespace rust_compiler {

class TypeBuilder {
  mlir::OpBuilder *builder;
  Target *target;

public:
  TypeBuilder(mlir::OpBuilder *builder, Target *target)
      : builder(builder), target(target) {}
  mlir::Type getType(std::shared_ptr<ast::types::Type>);
};

} // namespace rust_compiler
