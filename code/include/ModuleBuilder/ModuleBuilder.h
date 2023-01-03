#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/Module.h"
#include "AST/Statement.h"
#include "Mir/MirDialect.h"
#include "Target.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
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

  void build(std::shared_ptr<ast::Module> m, Target &target);

private:
  mlir::func::FuncOp buildFun(std::shared_ptr<ast::Function> f);
  mlir::func::FuncOp buildFunctionSignature(ast::FunctionSignature sig,
                                            mlir::Location locaction);
  std::optional<mlir::Value>
  emitBlockExpression(std::shared_ptr<ast::BlockExpression> blk);

  std::optional<mlir::Value>
  emitStatement(std::shared_ptr<ast::Statement> stmt);
  void buildLetStatement(std::shared_ptr<ast::LetStatement> letStmt);

  mlir::Value emitExpression(std::shared_ptr<ast::Expression> expr);

  mlir::Value
  emitExpressionWithBlock(std::shared_ptr<ast::ExpressionWithBlock> expr);
  mlir::Value buildExpressionWithoutBlock(
      std::shared_ptr<ast::ExpressionWithoutBlock> expr);

  void buildExpressionStatement(std::shared_ptr<ast::ExpressionStatement> expr);

  void buildItem(std::shared_ptr<ast::Item> item);

  mlir::Value emitArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression> expr);

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  //  mlir::LogicalResult declare(VarDeclExprAST &var, mlir::Value value);

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location getLocation(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.getFileName()),
                                     loc.getLineNumber(),
                                     loc.getColumnNumber());
  }
};

} // namespace rust_compiler
