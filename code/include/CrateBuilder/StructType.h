#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <string>

namespace rust_compiler::crate_builder {

class StructType {
  ast::StructStruct *definition;
  mlir::MemRefType memRef;
  mlir::DynamicType type;
  std::string typeName;

public:
  StructType(ast::StructStruct *definition, mlir::MemRefType memRef,
             mlir::DynamicType type, std::string typeName)
      : definition(definition), memRef(memRef), type(type), typeName(typeName) {
  }

  ast::StructStruct *getDefinition() const { return definition; }
};

} // namespace rust_compiler::crate_builder
