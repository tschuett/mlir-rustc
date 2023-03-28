#pragma once

#include "Analysis/Loops.h"
#include "Analysis/MemorySSA/MemorySSA.h"
#include "Analysis/ScalarEvolution.h"

#include <vector>

namespace rust_compiler::optimizer {

using namespace rust_compiler::analysis;

class LoopPlan {};

class LoopRecipe {};

class NoopPlan : public LoopPlan {};

class LoopPlanner {
  std::vector<analysis::LoopNest> &nest;
  analysis::ScalarEvolution *scev;
  MemorySSA *memorySSA;

public:
  LoopPlanner(std::vector<analysis::LoopNest> &nest,
              analysis::ScalarEvolution *scev, MemorySSA *memorySSA)
      : nest(nest), scev(scev), memorySSA(memorySSA) {}

  void run();

private:
  void plan(LoopNest &);
};

} // namespace rust_compiler::optimizer
