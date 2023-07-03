#pragma once

#include "ADT/ScopedHashTable.h"
#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/ArrayExpression.h"
#include "AST/BlockExpression.h"
#include "AST/CallExpression.h"
#include "AST/ComparisonExpression.h"
#include "AST/Crate.h"
#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/Function.h"
#include "AST/IfExpression.h"
#include "AST/IfLetExpression.h"
#include "AST/ItemDeclaration.h"
#include "AST/IteratorLoopExpression.h"
#include "AST/LetStatement.h"
#include "AST/LiteralExpression.h"
#include "AST/LoopExpression.h"
#include "AST/MethodCallExpression.h"
#include "AST/OperatorExpression.h"
#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/QualifiedPathInExpression.h"
#include "AST/ReturnExpression.h"
#include "AST/Statement.h"
#include "AST/StructExpression.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypePath.h"
#include "AST/VariableDeclaration.h"
#include "Basic/Ids.h"
#include "CrateBuilder/Target.h"
#include "Hir/HirDialect.h"
#include "Session/Session.h"
#include "TyCtx/TyCtx.h"

#include <cstdint>
#include <llvm/ADT/ScopedHashTable.h>
// #include <llvm/MC/TargetRegistry.h>
#include <llvm/Remarks/YAMLRemarkSerializer.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <optional>
#include <stack>

namespace rust_compiler::ast {
class PathExpression;
class PathInExpression;
class QualifiedPathInExpression;
class IteratorLoopExpression;
} // namespace rust_compiler::ast

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
  mlir::Value emitComparisonExpression(ast::ComparisonExpression *expr);

  mlir::Value emitCallExpression(ast::CallExpression *expr);
  mlir::Value emitMethodCallExpression(ast::MethodCallExpression *expr);
  void emitReturnExpression(ast::ReturnExpression *expr);
  mlir::Value emitPathExpression(ast::PathExpression *expr);
  mlir::Value emitPathInExpression(ast::PathInExpression *expr);
  mlir::Value
  emitQualifiedPathInExpression(ast::QualifiedPathInExpression *expr);
  mlir::Value emitIfExpression(ast::IfExpression *expr);
  mlir::Value emitIfLetExpression(ast::IfLetExpression *expr);
  mlir::Value emitComparePatternWithExpression(ast::patterns::Pattern *pattern,
                                               ast::Expression *expr);
  mlir::Value
  emitComparePatternWithOperatorExpression(ast::patterns::Pattern *pattern,
                                           ast::OperatorExpression *expr);
  mlir::Value emitMatchIfLetPattern(ast::patterns::Pattern *pattern,
                                    ast::Expression *expr);
  mlir::Value emitMatchIfLetNoTopAlt(ast::patterns::PatternNoTopAlt *pattern,
                                     ast::Expression *expr);
  mlir::Value emitLiteralExpression(ast::LiteralExpression *);
  mlir::Value emitArrayExpression(ast::ArrayExpression *array);
  mlir::Value emitStructExpression(ast::StructExpression *stru);
  mlir::Value emitIteratorLoopExpression(ast::IteratorLoopExpression *loop);
  mlir::Value emitTupleStructConstructor(ast::CallExpression*expr);

  mlir::FunctionType getFunctionType(ast::Function *);

  mlir::Type getType(ast::types::TypeExpression *);
  mlir::Type getExpression(ast::Expression *);
  mlir::Type getTypeNoBounds(ast::types::TypeNoBounds *);
  mlir::Type getTypePath(ast::types::TypePath *);

  mlir::Type convertTyTyToMLIR(tyctx::TyTy::BaseType *);
  mlir::MemRefType getMemRefType(tyctx::TyTy::BaseType *);

  void declare(basic::NodeId, mlir::Value);

  /// Helper conversion for a Rust AST location to an MLIR location.
  mlir::Location getLocation(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(loc.getFileName()),
                                     loc.getLineNumber(),
                                     loc.getColumnNumber());
  }

  uint64_t foldAsUsizeExpression(ast::Expression *);
  uint64_t foldAsUsizeWithoutBlock(ast::ExpressionWithoutBlock *);
  uint64_t foldAsUsizeWithBlock(ast::ExpressionWithBlock *);
  uint64_t foldAsUsizePathExpression(ast::PathExpression *);
  uint64_t foldAsUsizeItem(ast::Item *);
  uint64_t foldAsUsizeVisItem(ast::VisItem *);
  uint64_t foldAsUsizeConstantItem(ast::ConstantItem *);
  uint64_t foldAsUsizeLiteralExpression(ast::LiteralExpression *);

  ast::Crate *crate;

  enum class OwnerKind { Expression, Statement, Item };
  class Owner {
    ast::Expression *expr;
    ast::Statement *stmt;
    ast::Item *item;
    OwnerKind kind;

  public:
    OwnerKind getKind() const { return kind; }

    ast::Item *getItem() const { return item; }

    static Owner expression(ast::Expression *expression) {
      Owner owner;
      owner.kind = OwnerKind::Expression;
      owner.expr = expression;
      return owner;
    }

    static Owner statement(ast::Statement *stmt) {
      Owner owner;
      owner.kind = OwnerKind::Statement;
      owner.stmt = stmt;
      return owner;
    }

    static Owner Item(ast::Item *item) {
      Owner owner;
      owner.kind = OwnerKind::Item;
      owner.item = item;
      return owner;
    }
  };

  std::optional<Owner> getOwner(basic::NodeId id);
  std::optional<Owner> getOwnerCrate(basic::NodeId id);
  std::optional<Owner> getOwnerItem(basic::NodeId id, ast::Item *);
  std::optional<Owner> getOwnerVisItem(basic::NodeId id, ast::VisItem *);
  std::optional<Owner> getOwnerFunction(basic::NodeId id, ast::Function *);
  std::optional<Owner> getOwnerExpression(basic::NodeId id, ast::Expression *);
  std::optional<Owner> getOwnerExpressionWithBlock(basic::NodeId id,
                                                   ast::ExpressionWithBlock *);
  std::optional<Owner>
  getOwnerExpressionWithoutBlock(basic::NodeId id,
                                 ast::ExpressionWithoutBlock *);
  std::optional<Owner> getOwnerBlockExpression(basic::NodeId id,
                                               ast::BlockExpression *);
  std::optional<Owner> getOwnerStatement(basic::NodeId id, ast::Statement *);
  std::optional<Owner> getOwnerItemDeclaration(basic::NodeId id,
                                               ast::ItemDeclaration *);

  bool isConstantExpression(ast::Expression *);
  bool isConstantExpressionWithBlock(ast::ExpressionWithBlock *);
  bool isConstantExpressionWithoutBlock(ast::ExpressionWithoutBlock *);

  std::optional<size_t> getNrOfIterationsOfIntoIterator(ast::Expression *);
  std::optional<size_t>
  getNrOfIterationsOfIntoIterator(ast::ExpressionWithBlock *);
  std::optional<size_t>
  getNrOfIterationsOfIntoIterator(ast::ExpressionWithoutBlock *);
  std::optional<size_t>
  getNrOfIterationsOfIntoIterator(ast::OperatorExpression *);

  std::map<basic::NodeId, Owner> owners;

  std::map<basic::NodeId, mlir::Value> variables;

  bool isLiteralExpression(ast::Expression *) const;
};

} // namespace rust_compiler::crate_builder
