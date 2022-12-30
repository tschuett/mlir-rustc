#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace rust_compiler::Mir {
class MethodRegistry {
public:
  bool registerMethod(mlir::TypeID typeID, llvm::StringRef methodName,
                      llvm::StringRef opName);

  llvm::Optional<llvm::StringRef> lookupMethod(::mlir::TypeID Type,
                                               llvm::StringRef Name) const;

private:
  llvm::DenseMap<std::pair<mlir::TypeID, llvm::StringRef>, llvm::StringRef>
      methods;
};
} // namespace rust_compiler::Mir

#include "MirDialect.h.inc"

#define GET_OP_CLASSES
#include "MirOps.h.inc"
