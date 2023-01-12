#pragma once

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/BlockExpression.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/ItemDeclaration.h"
#include "AST/LetStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/Module.h"
#include "AST/ReturnExpression.h"
#include "AST/Statement.h"
#include "AST/VariableDeclaration.h"
#include "Mir/MirDialect.h"
#include "ModuleBuilder/Target.h"
#include "Target.h"

#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
  Target *target;

  llvm::ScopedHashTable<llvm::StringRef,
                        std::pair<mlir::Value, ast::VariableDeclaration *>>
      symbolTable;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::func::FuncOp> functionMap;

public:
  ModuleBuilder(std::string_view moduleName, Target *target,
                llvm::raw_ostream &OS, mlir::MLIRContext &context)
      : moduleName(moduleName), builder(&context),
        serializer(OS, llvm::remarks::SerializerMode::Separate) {
    context.getOrLoadDialect<Mir::MirDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
  };

  void build(std::shared_ptr<ast::Module> m);

private:
  mlir::func::FuncOp emitFun(std::shared_ptr<ast::Function> f);
  mlir::func::FuncOp buildFunctionSignature(ast::FunctionSignature sig,
                                            mlir::Location locaction);
  std::optional<mlir::Value>
  emitBlockExpression(std::shared_ptr<ast::BlockExpression> blk);

  std::optional<mlir::Value>
  emitStatement(std::shared_ptr<ast::Statement> stmt);
  void buildLetStatement(std::shared_ptr<ast::LetStatement> letStmt);

  std::optional<mlir::Value>
  emitStatements(std::shared_ptr<ast::Statements> stmts);

  mlir::Value emitExpression(std::shared_ptr<ast::Expression> expr);

  mlir::Value
  emitExpressionWithBlock(std::shared_ptr<ast::ExpressionWithBlock> expr);
  mlir::Value
  emitExpressionWithoutBlock(std::shared_ptr<ast::ExpressionWithoutBlock> expr);

  void buildExpressionStatement(std::shared_ptr<ast::ExpressionStatement> expr);

  // void buildItem(std::shared_ptr<ast::Item> item);
  void emitItemDeclaration(std::shared_ptr<ast::ItemDeclaration> item);

  mlir::Value emitArithmeticOrLogicalExpression(
      std::shared_ptr<ast::ArithmeticOrLogicalExpression> expr);

  mlir::Value
  emitOperatorExpression(std::shared_ptr<ast::OperatorExpression> opr);

  void emitItem(std::shared_ptr<ast::Item> item);
  void emitModule(std::shared_ptr<ast::Module> module);

  mlir::Value
  emitLiteralExpression(std::shared_ptr<ast::LiteralExpression> lit);

  void emitReturnExpression(std::shared_ptr<ast::ReturnExpression> ret);
  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(ast::VariableDeclaration &var, mlir::Value value);

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location getLocation(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.getFileName()),
                                     loc.getLineNumber(),
                                     loc.getColumnNumber());
  }
  mlir::Type getType(std::shared_ptr<ast::types::Type>);
};

} // namespace rust_compiler
