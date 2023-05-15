#pragma once

#include "Analysis/Loops.h"
#include "Analysis/MemorySSA/MemorySSA.h"
#include "Analysis/ScalarEvolution.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <string>
#include <vector>

namespace rust_compiler::optimizer {

using namespace rust_compiler::analysis;

//  LoopRotatePass or LoopUnrollAndJamPass
// LoopUnroll, LoopPeeling, LoopFlatten
// Data Dependency Analysis
/// The loop planer.
///
///
///
/// The cannonical 5 nested for-loops
///
/// \code{.cpp}
///
/// for( unknown ) {
///   for( unknown ) {
///     for( huge ) {
///       for( unknown ) {
///         for( small ) {
///         }
///         for( huge ) {
///         }
///       }
///     }
///     for( huge ) {
///       for( unknown ) {
///         for( small ) {
///         }
///         for( huge ) {
///         }
///       }
///     }
///   }
/// }
///
///\endcode

class BlockBase;

class Region;

class LoopPlan {
  //  BlockBase *enry;
};

class NoopPlan : public LoopPlan {};

// Parent of Region and BasicBlock
class BlockBase {
  std::string name;
};

// Single entry; single exit
class Region : public BlockBase {
  //  BlockBase *entry;
  //  BlockBase *exit;
  llvm::SmallPtrSet<BlockBase, 8> blocks;
};

class BasicBlock : public BlockBase {};

class LoopPlanner {
  std::vector<analysis::LoopNest> &nest;
  [[maybe_unused]] analysis::ScalarEvolution *scev;
  [[maybe_unused]] MemorySSA *memorySSA;

public:
  LoopPlanner(std::vector<analysis::LoopNest> &nest,
              analysis::ScalarEvolution *scev, MemorySSA *memorySSA)
      : nest(nest), scev(scev), memorySSA(memorySSA) {}

  void run();

private:
  void plan(LoopNest &);
};

} // namespace rust_compiler::optimizer

// https://reviews.llvm.org/D28975
