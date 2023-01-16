#include "ModuleBuilder/ModuleBuilder.h"

#include "AST/Types/PrimitiveTypes.h"
#include "AST/Types/Types.h"

namespace rust_compiler {

mlir::Type ModuleBuilder::getType(std::shared_ptr<ast::types::Type> type) {
  switch (type->getKind()) {
  case ast::types::TypeKind::PrimitiveType: {
    auto prim = std::static_pointer_cast<ast::types::PrimitiveType>(type);
    switch (prim->getKind()) {
    case ast::types::PrimitiveTypeKind::Usize: {
      return builder.getIntegerType(64, false);
    }
    case ast::types::PrimitiveTypeKind::Isize: {
      return builder.getIntegerType(64, true);
    }
    case ast::types::PrimitiveTypeKind::U128: {
      return builder.getIntegerType(128, false);
    }
    case ast::types::PrimitiveTypeKind::I128: {
      return builder.getIntegerType(128, true);
    }
    case ast::types::PrimitiveTypeKind::U64: {
      return builder.getIntegerType(64, false);
    }
    case ast::types::PrimitiveTypeKind::I64: {
      return builder.getIntegerType(64, true);
    }
    case ast::types::PrimitiveTypeKind::U32: {
      return builder.getIntegerType(32, false);
    }
    case ast::types::PrimitiveTypeKind::I32: {
      return builder.getIntegerType(32, true);
    }
    case ast::types::PrimitiveTypeKind::U16: {
      return builder.getIntegerType(16, false);
    }
    case ast::types::PrimitiveTypeKind::I16: {
      return builder.getIntegerType(16, true);
    }
    case ast::types::PrimitiveTypeKind::U8: {
      return builder.getIntegerType(8, false);
    }
    case ast::types::PrimitiveTypeKind::I8: {
      return builder.getIntegerType(8, true);
    }
    case ast::types::PrimitiveTypeKind::Boolean: {
      return builder.getIntegerType(1, false);
    }
    case ast::types::PrimitiveTypeKind::F64: {
      return builder.getF64Type();
    }
    case ast::types::PrimitiveTypeKind::F32: {
      return builder.getF32Type();
    }
    case ast::types::PrimitiveTypeKind::Binary64: {
      return builder.getF64Type();
    }
    case ast::types::PrimitiveTypeKind::Binary32: {
      return builder.getF32Type();
    }
    default: {
      llvm::outs() << "unknown type"
                   << "\n";
      exit(EXIT_FAILURE);
    }
    }
  }
  };
}

} // namespace rust_compiler


// Usize and Isize depend on Target!
