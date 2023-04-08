#pragma once

#include "ADT/ScopedHashTable.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/BlockExpression.h"
#include "AST/CallExpression.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/Function.h"
#include "AST/LetStatement.h"
#include "AST/LoopExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/Types/TypeExpression.h"
#include "AST/VariableDeclaration.h"
#include "CrateBuilder/Target.h"
#include "Hir/HirDialect.h"
#include "AST/ReturnExpression.h"

#include <llvm/MC/TargetRegistry.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <optional>
#include <stack>

namespace rust_compiler::crate_builder {

/// Lowers an AST Crate into a mix of Hir, Arith, MemRef, ... dialects into an
/// MLIR module
class CrateBuilder {
  std::string moduleName;
  // mlir::MLIRContext context;
  // mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::OpBuilder builder;
  mlir::ModuleOp &theModule;
  llvm::remarks::YAMLRemarkSerializer serializer;
  Target target;
  std::stack<mlir::Block *> breakPoints;

  adt::ScopedHashTable<std::string, mlir::Value> symbolTable;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::func::FuncOp> functionMap;

  mlir::Region *functionRegion = nullptr;
  mlir::Block *entryBlock = nullptr;

public:
  CrateBuilder(llvm::raw_ostream &OS, mlir::ModuleOp &theModule,
               mlir::MLIRContext &context, llvm::TargetMachine *tm)
      : builder(&context), theModule(theModule),
        serializer(OS, llvm::remarks::SerializerMode::Separate),
        target(tm){

            //    // Create `Target`
            //    std::string theTriple =
            //        llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());
            //
            //    std::string error;
            //    const llvm::Target *theTarget =
            //        llvm::TargetRegistry::lookupTarget(theTriple, error);
            //
            //    std::string featuresStr;
            //    std::string cpu = std::string(llvm::sys::getHostCPUName());
            //    std::unique_ptr<::llvm::TargetMachine> tm;
            //    tm.reset(theTarget->createTargetMachine(
            //        theTriple, /*CPU=*/cpu,
            //        /*Features=*/featuresStr, llvm::TargetOptions(),
            //        /*Reloc::Model=*/llvm::Reloc::Model::PIC_,
            //        /*CodeModel::Model=*/std::nullopt,
            //        llvm::CodeGenOpt::Aggressive));
            //    assert(tm && "Failed to create TargetMachine");
            //
            //    Target target = {tm.get()};
            //
            //    // builder = {&context};
            //
            //    context.getOrLoadDialect<hir::HirDialect>();
            //    context.getOrLoadDialect<mlir::func::FuncDialect>();
            //    context.getOrLoadDialect<mlir::arith::ArithDialect>();
            //    //context.getOrLoadDialect<mlir::async::AsyncDialect>();
            //    //context.getOrLoadDialect<mlir::memref::MemRefDialect>();
            //    context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
            //
            //    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
        };

  void emitCrate(rust_compiler::ast::Crate *crate);

  mlir::ModuleOp getModule() const { return theModule; };

private:
  void emitItem(std::shared_ptr<ast::Item> &);
  void emitVisItem(std::shared_ptr<ast::VisItem>);
  void emitFunction(std::shared_ptr<ast::Function>);
  void emitModule(std::shared_ptr<ast::Module>);
  std::optional<mlir::Value> emitBlockExpression(ast::BlockExpression *);
  std::optional<mlir::Value> emitStatements(ast::Statements);
  mlir::Value emitExpression(ast::Expression* expr);
  mlir::Value emitExpressionWithoutBlock(ast::ExpressionWithoutBlock* expr);
  mlir::Value emitExpressionWithBlock(ast::ExpressionWithBlock* expr);
  void emitStatement(ast::Statement *);
  void emitExpressionStatement(ast::ExpressionStatement *stmt);
  void emitLetStatement(ast::LetStatement *stmt);
  mlir::Value emitLoopExpression(ast::LoopExpression* expr);
  mlir::Value
  emitOperatorExpression(ast::OperatorExpression* expr);
  mlir::Value emitArithmeticOrLogicalExpression(
      ast::ArithmeticOrLogicalExpression* expr);

  mlir::Value emitCallExpression(ast::CallExpression* expr);
  mlir::Value
  emitMethodCallExpression(ast::MethodCallExpression* expr);
  mlir::Value
  emitReturnExpression(ast::ReturnExpression* expr);

  mlir::FunctionType getFunctionType(ast::Function *);

  mlir::Type getType(ast::types::TypeExpression *);

  /// Helper conversion for a Rust AST location to an MLIR location.
  mlir::Location getLocation(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.getFileName()),
                                     loc.getLineNumber(),
                                     loc.getColumnNumber());
  }
};

} // namespace rust_compiler::crate_builder
