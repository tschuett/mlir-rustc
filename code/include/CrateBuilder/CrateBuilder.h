#pragma once

#include "ADT/ScopedHashTable.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/BlockExpression.h"
#include "AST/CallExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/Function.h"
#include "AST/LetStatement.h"
#include "AST/LoopExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "AST/VariableDeclaration.h"
#include "CrateBuilder/Target.h"
#include "Hir/HirDialect.h"
#include "Session/Session.h"
#include "TyCtx/TyCtx.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Target/TargetMachine.h>
//#include <llvm/TargetParser/Host.h>
//#include <mlir/Dialect/Arith/IR/Arith.h>
//#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <optional>
#include <stack>

namespace rust_compiler::ast {
class PathExpression;
}

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

  llvm::ScopedHashTable<basic::NodeId, mlir::Value> symbolTable;

  llvm::ScopedHashTable<basic::NodeId, mlir::Value> allocaTable;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::func::FuncOp> functionMap;

  mlir::Region *functionRegion = nullptr;
  mlir::Block *entryBlock = nullptr;

  tyctx::TyCtx *tyCtx;

public:
  CrateBuilder(llvm::raw_ostream &OS, mlir::ModuleOp &theModule,
               mlir::MLIRContext &context, llvm::TargetMachine *tm)
      : builder(&context), theModule(theModule),
        serializer(OS, llvm::remarks::SerializerMode::Separate), target(tm) {
    tyCtx = rust_compiler::session::session->getTypeContext();

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
  void emitItem(ast::Item *item);
  void emitVisItem(ast::VisItem *vis);
  void emitFunction(ast::Function *fun);
  void emitModule(ast::Module *);
  std::optional<mlir::Value> emitBlockExpression(ast::BlockExpression *);
  std::optional<mlir::Value> emitStatements(ast::Statements);
  std::optional<mlir::Value> emitExpression(ast::Expression *expr);
  std::optional<mlir::Value>
  emitExpressionWithoutBlock(ast::ExpressionWithoutBlock *expr);
  mlir::Value emitExpressionWithBlock(ast::ExpressionWithBlock *expr);
  void emitStatement(ast::Statement *);
  void emitExpressionStatement(ast::ExpressionStatement *stmt);
  void emitLetStatement(ast::LetStatement *stmt);
  mlir::Value emitLoopExpression(ast::LoopExpression *expr);
  mlir::Value emitOperatorExpression(ast::OperatorExpression *expr);
  mlir::Value
  emitArithmeticOrLogicalExpression(ast::ArithmeticOrLogicalExpression *expr);
  mlir::Value
  emitComparisonExpression(ast::ComparisonExpression *expr);

  mlir::Value emitCallExpression(ast::CallExpression *expr);
  mlir::Value emitMethodCallExpression(ast::MethodCallExpression *expr);
  void emitReturnExpression(ast::ReturnExpression *expr);
  mlir::Value emitPathExpression(ast::PathExpression *expr);

  mlir::FunctionType getFunctionType(ast::Function *);

  mlir::Type getType(ast::types::TypeExpression *);
  mlir::Type getExpression(ast::Expression *);
  mlir::Type getTypeNoBounds(ast::types::TypeNoBounds *);
  mlir::Type getTypePath(ast::types::TypePath *);

  mlir::Type convertTyTyToMLIR(tyctx::TyTy::BaseType *);

  void declare(basic::NodeId, mlir::Value);

  /// Helper conversion for a Rust AST location to an MLIR location.
  mlir::Location getLocation(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.getFileName()),
                                     loc.getLineNumber(),
                                     loc.getColumnNumber());
  }
};

} // namespace rust_compiler::crate_builder
