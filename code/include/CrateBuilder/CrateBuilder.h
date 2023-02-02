#pragma once

#include "ADT/ScopedHashTable.h"
#include "AST/Crate.h"
#include "AST/VariableDeclaration.h"
#include "CrateBuilder/Target.h"
#include "Hir/HirDialect.h"

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Support/Host.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <stack>

namespace rust_compiler::crate_builder {

class CrateBuilder {
  std::string moduleName;
  // mlir::MLIRContext context;
  // mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder;
  mlir::ModuleOp theModule;
  llvm::remarks::YAMLRemarkSerializer serializer;
  Target *target;
  std::stack<mlir::Block *> breakPoints;

  adt::ScopedHashTable<std::string,
                       std::pair<mlir::Value, ast::VariableDeclaration *>>
      symbolTable;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::func::FuncOp> functionMap;

  mlir::Region *functionRegion = nullptr;
  mlir::Block *entryBlock = nullptr;

public:
  CrateBuilder(llvm::raw_ostream &OS, mlir::MLIRContext &context)
      : builder(&context),
        serializer(OS, llvm::remarks::SerializerMode::Separate) {

    // Create `Target`
    std::string theTriple =
        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

    std::string error;
    const llvm::Target *theTarget =
        llvm::TargetRegistry::lookupTarget(theTriple, error);

    std::string featuresStr;
    std::string cpu = "sapphirerapids";
    cpu = llvm::sys::getHostCPUName();
    std::unique_ptr<::llvm::TargetMachine> tm;
    tm.reset(theTarget->createTargetMachine(
        theTriple, /*CPU=*/cpu,
        /*Features=*/featuresStr, llvm::TargetOptions(),
        /*Reloc::Model=*/llvm::Reloc::Model::PIC_,
        /*CodeModel::Model=*/std::nullopt, llvm::CodeGenOpt::Aggressive));
    assert(tm && "Failed to create TargetMachine");

    Target target = {tm.get()};

    //builder = {&context};

    context.getOrLoadDialect<hir::HirDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    // context.getOrLoadDialect<mlir::async::AsyncDialect>();
    // context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();

    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  };

  void emitCrate(rust_compiler::ast::Crate &crate);

  mlir::ModuleOp getModule() const { return theModule; };

private:
  void emitItem(std::unique_ptr<ast::Item>&);
  void emitVisItem(std::shared_ptr<ast::VisItem>);
  void emitFunction(std::shared_ptr<ast::Function>);
  void emitModule(std::shared_ptr<ast::Module>);
};

void build(rust_compiler::ast::Crate &crate);

} // namespace rust_compiler::crate_builder
