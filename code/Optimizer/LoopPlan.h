#pragma once

#include "Analysis/Loops.h"

namespace rust_compiler::optimizer {

class LoopPlan {};

class LoopRecipe {};

class DoNothingPlan : public LoopPlan {};

class LoopPlanner {
  analysis::LoopNest *nest;

public:
  LoopPlanner(analysis::LoopNest *nest);
};

} // namespace rust_compiler::optimizer
