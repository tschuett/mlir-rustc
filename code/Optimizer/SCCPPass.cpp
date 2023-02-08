#include "Mir/MirOps.h"
#include "Optimizer/ComputeKnownBits.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_SCCPPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class SCCPPass : public rust_compiler::optimizer::impl::SCCPPassBase<SCCPPass> {
public:
  void runOnOperation() override;
};

using namespace llvm;
using namespace mlir;

class LatticeValue {
  enum class LatticeValueKind { Unknown, Constant, Overdefined };

public:
  LatticeValue()
      : value(nullptr), kind(LatticeValueKind::Unknown), dialect(nullptr) {}
  LatticeValue(Attribute attr, Dialect *dialect)
      : value(attr), kind(LatticeValueKind::Constant), dialect(dialect) {}

  /// Mark the lattice value as overdefined.
  void markOverdefined() {
    kind = LatticeValueKind::Overdefined;
    value = nullptr;
    dialect = nullptr;
  }

  /// Returns true if this lattice value is unknown.
  bool isUnknown() const { return kind == LatticeValueKind::Unknown; }

  /// Returns true if the lattice is overdefined.
  bool isOverdefined() const { return kind == LatticeValueKind::Overdefined; }

  /// If this lattice is constant, return the constant. Returns nullptr
  /// otherwise.
  Attribute getConstant() const {
    if (kind == LatticeValueKind::Constant)
      return value;
    return nullptr;
  }

  /// If this lattice is constant, return the dialect to use when materializing
  /// the constant.
  Dialect *getConstantDialect() const {
    assert(getConstant() && "expected valid constant");
    return dialect;
  }

  // merge the value rgs l
  bool meet(const LatticeValue &rhs) {
    if (isOverdefined() || rhs.isUnknown())
      return false;
    // if we are unknown, just take the value of rhs
    if (isUnknown()) {
      value = rhs.value;
      kind = rhs.kind;
      dialect = rhs.dialect;
      return true;
    }

    // Otherwise
    if (value == rhs.value and kind == rhs.kind) {
      markOverdefined();
      return true;
    }
    return false;
  };

private:
  Attribute value;
  LatticeValueKind kind;
  Dialect *dialect;
};

class SCCPSolver {
public:
  /// Initialize the solver with a given set of regions.
  SCCPSolver(mlir::ModuleOp *module);

  /// Run the solver until it converges.
  void solve();

  /// Rewrite the given regions using the computing analysis. This replaces the
  /// uses of all values that have been computed to be constant, and erases as
  /// many newly dead operations.
  void rewrite(MLIRContext *context,
               llvm::MutableArrayRef<mlir::Region> regions);

private:
  /// Visit the given operation and compute any necessary lattice state.
  void visitOperation(Operation *op);

  /// Visit the given operation, which defines regions, and compute any
  /// necessary lattice state. This also resolves the lattice state of both the
  /// operation results and any nested regions.
  void visitRegionOperation(Operation *op);

  /// Visit the given terminator operation and compute any necessary lattice
  /// state.
  void visitTerminatorOperation(Operation *op,
                                ArrayRef<Attribute> constantOperands);

  /// Visit the given block and compute any necessary lattice state.
  void visitBlock(Block *block);

  /// Visit argument #'i' of the given block and compute any necessary lattice
  /// state.
  void visitBlockArgument(Block *block, int i);

  /// Mark the given block as executable. Returns false if the block was already
  /// marked executable.
  bool markBlockExecutable(Block *block);

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markOverdefined(Value value);

  /// Mark all of the given values as overdefined.
  template <typename ValuesT> void markAllOverdefined(ValuesT values) {
    for (auto value : values)
      markOverdefined(value);
  }

  template <typename ValuesT>
  void markAllOverdefined(Operation *op, ValuesT values) {
    markAllOverdefined(values);
    opWorklist.push_back(op);
  }

  /// Returns true if the given value was marked as overdefined.
  bool isOverdefined(Value value) const;

  /// Merge in the given lattice 'from' into the lattice 'to'. 'owner'
  /// corresponds to the parent operation of 'to'.
  void meet(Operation *owner, LatticeValue &to, const LatticeValue &from);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const;

  /// The lattice for each SSA value.
  llvm::DenseMap<mlir::Value, LatticeValue> latticeValues;

  /// The set of blocks that are known to execute, or are intrinsically live.
  llvm::SmallPtrSet<mlir::Block *, 16> executableBlocks;

  /// The set of control flow edges that are known to execute.
  llvm::DenseSet<std::pair<mlir::Block *, mlir::Block *>> executableEdges;

  /// A worklist containing blocks that need to be processed.
  llvm::SmallVector<mlir::Block *, 64> blockWorklist;

  /// A worklist of operations that need to be processed.
  llvm::SmallVector<mlir::Operation *, 64> opWorklist;
};
} // namespace

/// Initialize the solver with a given set of regions.
SCCPSolver::SCCPSolver(mlir::ModuleOp *module) {
  module->walk([&](mlir::func::FuncOp f) {
    if (not f.empty()) {
      Block *entryBlock = &f.front();

      // Mark the entry block as executable.
      markBlockExecutable(entryBlock);

      // The values passed to these regions are invisible, so mark any arguments
      // as overdefined.
      markAllOverdefined(entryBlock->getArguments());
    }
  });
}

void SCCPSolver::solve() {
  while (!blockWorklist.empty() || !opWorklist.empty()) {
    // Process any operations in the op worklist.
    while (!opWorklist.empty()) {
      Operation *op = opWorklist.pop_back_val();

      // Visit all of the live users to propagate changes to this operation.
      for (Operation *user : op->getUsers()) {
        if (isBlockExecutable(user->getBlock()))
          visitOperation(user);
      }
    }

    // Process any blocks in the block worklist.
    while (!blockWorklist.empty())
      visitBlock(blockWorklist.pop_back_val());
  }
}

bool SCCPSolver::isBlockExecutable(Block *block) const {
  return executableBlocks.count(block);
}

bool SCCPSolver::markBlockExecutable(Block *block) {
  bool marked = executableBlocks.insert(block).second;
  if (marked)
    blockWorklist.push_back(block);
  return marked;
}

void SCCPSolver::markOverdefined(Value value) {
  latticeValues[value].markOverdefined();
}

void SCCPSolver::visitOperation(Operation *op) {
  // Collect all of the constant operands feeding into this operation. If any
  // are not ready to be resolved, bail out and wait for them to resolve.
  SmallVector<Attribute, 8> operandConstants;
  operandConstants.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    // Make sure all of the operands are resolved first.
    auto &operandLattice = latticeValues[operand];
    if (operandLattice.isUnknown())
      return;
    operandConstants.push_back(operandLattice.getConstant());
  }

  // If this is a terminator operation, process any control flow lattice state.
  if (op->isKnownTerminator())
    visitTerminatorOperation(op, operandConstants);

  // Process region holding operations. The region visitor processes result
  // values, so we can exit afterwards.
  if (op->getNumRegions())
    return visitRegionOperation(op);

  // If this op produces no results, it can't produce any constants.
  if (op->getNumResults() == 0)
    return;

  // If all of the results of this operation are already overdefined, bail out
  // early.
  auto isOverdefinedFn = [&](Value value) { return isOverdefined(value); };
  if (llvm::all_of(op->getResults(), isOverdefinedFn))
    return;

  // Save the original operands and attributes just in case the operation folds
  // in-place. The constant passed in may not correspond to the real runtime
  // value, so in-place updates are not allowed.
  SmallVector<Value, 8> originalOperands(op->getOperands());
  NamedAttributeList originalAttrs = op->getAttrList();

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(operandConstants, foldResults)))
    return markAllOverdefined(op, op->getResults());

  if (failed(op->fold(operandConstants, foldResults)))
    return markAllOverdefined(op, op->getResults());

  // If the folding was in-place, mark the results as overdefined and reset the
  // operation. We don't allow in-place folds as the desire here is for
  // simulated execution, and not general folding.
  if (foldResults.empty()) {
    op->setOperands(originalOperands);
    op->setAttrs(originalAttrs);
    return markAllOverdefined(op, op->getResults());

    // Merge the fold results into the lattice for this operation.
    assert(foldResults.size() == op->getNumResults() && "invalid result size");
    Dialect *opDialect = op->getDialect();
    for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
      LatticeValue &resultLattice = latticeValues[op->getResult(i)];

      // Merge in the result of the fold, either a constant or a value.
      OpFoldResult foldResult = foldResults[i];
      if (Attribute foldAttr = foldResult.dyn_cast<Attribute>())
        meet(op, resultLattice, LatticeValue(foldAttr, opDialect));
      else
        meet(op, resultLattice, latticeValues[foldResult.get<Value>()]);
    }
  }
}

void SCCPSolver::visitBlock(Block *block) {
  // If the block is not the entry block we need to compute the lattice state
  // for the block arguments. Entry block argument lattices are computed
  // elsewhere, such as when visiting the parent operation.
  if (!block->isEntryBlock()) {
    for (int i : llvm::seq<int>(0, block->getNumArguments()))
      visitBlockArgument(block, i);
  }

  // Visit all of the operations within the block.
  for (Operation &op : *block)
    visitOperation(&op);
}

void SCCPSolver::visitRegionOperation(Operation *op) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    Block *entryBlock = &region.front();
    markBlockExecutable(entryBlock);
    markAllOverdefined(entryBlock->getArguments());
  }
}

void SCCPSolver::visitTerminatorOperation(
    Operation *op, ArrayRef<Attribute> constantOperands) {
  if (op->getNumSuccessors() == 0)
    return;

  // Try to resolve to a specific successor with the constant operands.
  if (auto branch = dyn_cast<BranchOpInterface>(op)) {
    if (Block *singleSucc = branch.getSuccessorForOperands(constantOperands)) {
      markEdgeExecutable(op->getBlock(), singleSucc);
      return;
    }
  }

  // Otherwise, conservatively treat all edges as executable.
  Block *block = op->getBlock();
  for (Block *succ : op->getSuccessors())
    markEdgeExecutable(block, succ);
}

void SCCPPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  SCCPSolver solver{&module};
  solver.solve();
  //  module.walk([&](mlir::func::FuncOp f) {
  //    //  isAsync() -> rewrite
  //    //  if (isa<rust_compiler::Mir::AwaitOp>(op)) {
  //    //  }
  //  });
}

// https://reviews.llvm.org/D78397
