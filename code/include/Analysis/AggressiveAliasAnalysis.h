#pragma once

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <optional>
#include <set>
#include <span>
#include <vector>

namespace rust_compiler::analysis {

/// Choi, Automatic construction of sparse data flow evaluation graphs.
class SparseEvaluationGraph {
public:
  mlir::func::FuncOp *getEntryNode() const { return entryNode; }
  mlir::func::FuncOp *getExitNode() const { return exitNode; }

private:
  llvm::SmallPtrSet<mlir::func::FuncOp *, 8> genNodes;
  llvm::SmallPtrSet<mlir::func::FuncOp *, 8> meetNodes;
  std::vector<std::pair<mlir::func::FuncOp *, mlir::func::FuncOp *>> edges;
  mlir::func::FuncOp *entryNode;
  mlir::func::FuncOp *exitNode;
};

/// A constant SSA value.
/// Inspired by Constant propagation analysis.
class ConstantValue {
public:
  explicit ConstantValue(mlir::Attribute constant, mlir::Dialect *dialect)
      : constant(constant), dialect(dialect) {}

  mlir::Dialect *getConstantDialect() const { return dialect; }

  bool isInitialized() const { return not constant.has_value(); }

  /// Compare the constant values.
  bool operator==(const ConstantValue &rhs) const {
    return constant == rhs.constant;
  }

private:
  std::optional<mlir::Attribute> constant;
  mlir::Dialect *dialect = nullptr;
};

/// An alias set is associated with a program point
class AliasSet {
public:
  // next hack
  AliasSet() = default;
  AliasSet getSubset(mlir::Value *) const;

private:
  AliasSet(std::set<std::pair<mlir::Value *, mlir::Value *>>);

  /// the mlir::Value s can be seen as addresses
  /// pairs of addresses may-alias
  std::set<std::pair<mlir::Value *, mlir::Value *>> mayAliases;
};

class Function {
public:
  AliasSet getEntryAliaSet(mlir::Value *);
  AliasSet getExitAliasSet(mlir::Value *);

private:
  AliasSet entryAliasSet;
  AliasSet exitAliasSet;
};

class ControlFlowGraph {
public:
  void addEdge(llvm::StringRef from, llvm::StringRef to, mlir::func::FuncOp *f);

  std::vector<mlir::func::FuncOp *> getFunctions() const { return funs; };

private:
  std::vector<mlir::func::FuncOp *> funs;
  llvm::StringMap<std::string> edges;
};

/// Efficient Flow-Sensitive of Pointer-Induced Interprocedural Computation
/// Aliases and Side Effects

/// Interprocedural Pointer Alias Analysis
/// Michael Hind, Michael Burke, Paul Carini, and Jong-Deok Choi
class AggressiveAliasAnalysis {
public:
  AggressiveAliasAnalysis(mlir::ModuleOp &mod);

  /// Given two values, return their aliasing behavior.
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

private:
  void initialize(mlir::ModuleOp &mod);
  void buildInitialCfg(mlir::ModuleOp &mod);
  void analyzeFunction(mlir::func::FuncOp *f);
  voin joinLoadOrStore(Function *f, mlir::Value);
  voin joinCallOp(Function *f); // FIXME

  ControlFlowGraph cfg;
  llvm::StringMap<std::unique_ptr<Function>> functions;
};

} // namespace rust_compiler::analysis
