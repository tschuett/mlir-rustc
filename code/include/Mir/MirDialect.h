#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

namespace rust_compiler::Mir {
class MethodRegistry {
public:
  /// Initializes this registry.
  void init(mlir::MLIRContext &Ctx);

  bool registerMethod(mlir::TypeID typeID, llvm::StringRef methodName,
                      llvm::StringRef opName);

  std::optional<llvm::StringRef> lookupMethod(::mlir::TypeID Type,
                                              llvm::StringRef Name) const;

  void registerDefinition(llvm::StringRef MethodName,
                          mlir::func::FuncOp Definition);

  std::optional<mlir::func::FuncOp>
  lookupDefinition(llvm::StringRef MethodName,
                   mlir::FunctionType FunctionType) const;

private:
  static constexpr llvm::StringLiteral ModuleName{"MirDefs"};

  llvm::DenseMap<std::pair<mlir::TypeID, llvm::StringRef>, llvm::StringRef>
      Methods;
  llvm::DenseMap<std::pair<llvm::SmallString<0>, mlir::FunctionType>,
                 mlir::func::FuncOp>
      Definitions;
  mlir::ModuleOp Module;
};

} // namespace rust_compiler::Mir

#include "MirDialect.h.inc"

#define GET_OP_CLASSES
#include "MirOps.h.inc"
