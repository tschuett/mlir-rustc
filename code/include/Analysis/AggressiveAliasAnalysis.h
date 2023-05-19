#pragma once

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Value.h>
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
/// Limitations: loads and stores may only generate one SSA
/// value. Only FuncOp dialect supported.

/// Choi, Automatic construction of sparse data flow evaluation graphs.
class SparseEvaluationGraph {
public:
  mlir::func::FuncOp *getEntryNode() const { return entryNode; }
  mlir::func::FuncOp *getExitNode() const { return exitNode; }

private:
  llvm::SmallPtrSet<mlir::func::FuncOp, 8> genNodes;
  llvm::SmallPtrSet<mlir::func::FuncOp, 8> meetNodes;
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

  mlir::Value getAddress() const { return address; }
  mlir::Value getStorage() const { return storage; }

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
  mlir::Value getRight() const;

private:
  mlir::Value left;
  mlir::Value right;
};

/// An alias set is associated with a program point. A set of pairs of
/// access paths that may alias.
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

class Function;

class CallSite {
  Function *fun;

public:
  Function *getFun() const { return fun; }

  bool isNull() const;

  friend bool operator==(const CallSite &l, const CallSite &r) {
    return l.fun == r.fun;
  }
};

/// Alias instance (AI)
class AliasInstance {
  AliasRelation ar;
  AliasSet sourceAliasSet; // SAS_c
  CallSite c;

public:
  Function *getFun() const { return c.getFun(); }

  /// AliasRelation
  void setAliasRelation(const AliasRelation &);
  AliasRelation getAliasRelation() const;

  /// AliasSet
  void setAliasSet(const AliasSet &);
  AliasSet getAliasSet() const;

  // CallSite
  void setCallSite(const CallSite &);
  CallSite getCallSite() const;
};

/// Alias instance set
class AliasInstanceSet {

public:
  AliasInstanceSet getSubset(mlir::Value) const;

  AliasInstanceSet minus(const AliasInstanceSet &);

  std::vector<AliasInstance> getSets() const;

  AliasInstanceSet join(const AliasInstanceSet &);

  std::vector<AliasInstance> getSetsWhereAliasRelationLeftIs(mlir::Value);

  void insert(const AliasInstance &);

private:
  std::set<AliasInstance> mayAliases;
};

class Function {
public:
  AliasInstanceSet getEntryAliasSet();
  AliasInstanceSet getExitAliasSet();

  void setExitSet(const AliasInstanceSet &);

  mlir::func::FuncOp getFuncOp();

private:
  AliasInstanceSet entryAliasInstanceSet;
  AliasInstanceSet exitAliasInstanceSet;
};

class ControlFlowGraph {
public:
  void addEdge(llvm::StringRef from, llvm::StringRef to, Function *f);

  std::vector<Function *> getFunctions() const { return funs; };

  std::vector<Function *> getCalleesOf(Function *) const;

private:
  std::vector<Function *> funs;
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
  bool analyzeFunction(Function *f);

  /// joins or transfer functions
  void transferFunLoad(Function *f, const Load &);
  void transferFunStore(Function *f, const Store &);
  void transferFunCallOp(Function *f); // FIXME

  Function *getFunction(mlir::func::FuncOp);
  ControlFlowGraph cfg;
  llvm::StringMap<std::unique_ptr<Function>> functions;
};

} // namespace rust_compiler::analysis
