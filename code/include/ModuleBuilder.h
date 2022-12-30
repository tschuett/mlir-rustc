#pragma once

#include "AST/Module.h"
#include "Mir/MirDialect.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <string_view>

namespace rust_compiler {

class ModuleBuilder {
  std::string moduleName;
  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  llvm::remarks::YAMLRemarkSerializer serializer;

  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

public:
  ModuleBuilder(std::string_view moduleName, llvm::raw_ostream &OS)
      : moduleName(moduleName), context(), builder(&context),
        serializer(OS, llvm::remarks::SerializerMode::Separate) {
    context.getOrLoadDialect<Mir::MirDialect>();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  };

  void build(std::shared_ptr<ast::Module> m);

private:
  Mir::FuncOp buildFun(std::shared_ptr<ast::Function> f);
  Mir::FuncOp buildFunctionSignature(ast::FunctionSignature sig);
};

} // namespace rust_compiler
