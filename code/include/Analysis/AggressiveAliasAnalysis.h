#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

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
#include <variant>
#include <vector>

namespace rust_compiler::analysis {

/// Efficient Flow-Sensitive of Pointer-Induced Interprocedural
/// Computation Aliases and Side Effects. Jong-Deok Choi, Michael
/// Burke, and Paul Carini.
///
/// Limitations: loads and stores may only generate one SSA value.

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

  // need to store callee names!
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

// enum class AccessKind { Load, Store };

class MemoryAccess {
  // AccessKind access;
  //  std::variant<ConstantValue, mlir::Value> address;
  // size_t size;
  // mlir::Value address;

private:
  //  MemoryAccess(AccessKind, mlir::Value address, mlir::OpResult result,
  //               mlir::MemRefType size);

  // bool mayAlias(mlir::Value) const;

  // mlir::Value getValue() const { return address; }
  //  hash hack
  // friend bool operator<(const MemoryAccess &l, const MemoryAccess &r) {
  //   return hash_value(l.address) < hash_value(r.address);
  // }
};

/// A load: address and storage value
class Load : public MemoryAccess {
public:
  Load(mlir::Value address, mlir::Value storage);

private:
  mlir::Value storage;
  mlir::Value address;
};

/// a store: a value and an address
class Store : public MemoryAccess {
public:
  Store(mlir::Value value, mlir::Value address);

  mlir::Value getAddress() const { return address; }
  mlir::Value getValue() const { return value; }

private:
  mlir::Value value;
  mlir::Value address;
};

/// aka AR
class AliasRelation {
public:
  AliasRelation replaceWith(mlir::Value, mlir::Value);

  mlir::Value getLeft() const;

private:
  mlir::Value left;
  mlir::Value right;
};

/// An alias set is associated with a program point. A set of pairs of
/// access path that may alias.
class AliasSet {
public:
  // next hack
  AliasSet() = default;

  AliasSet getSubset(mlir::Value) const;

  AliasSet minus(const AliasSet &);
  AliasSet join(const AliasSet &);
  AliasSet whereLeftIs(mlir::Value) const;

  void add(const AliasRelation &ar);

  std::vector<AliasRelation> getSets() const;

private:
  /// aliasing occurs when two or more l-value expressions reference
  /// the same storage location at the same program point p.
  std::set<AliasRelation> mayAliases;
};

class Function {
public:
  AliasSet getEntryAliasSet();
  AliasSet getExitAliasSet();

  void setExitSet(const AliasSet&);
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

/// Aiasing occurs when two or more l-value expressions reference the
/// same storage location at the same program point p.
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
  bool analyzeFunction(mlir::func::FuncOp *f);

  /// joins or transfer functions
  void transferFunLoad(Function *f, const Load &);
  void transferFunStore(Function *f, const Store &);
  void transferFunCallOp(Function *f); // FIXME

  ControlFlowGraph cfg;
  llvm::StringMap<std::unique_ptr<Function>> functions;
};

} // namespace rust_compiler::analysis
