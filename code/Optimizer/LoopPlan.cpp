#include "LoopPlan.h"

using namespace rust_compiler::analysis;

namespace rust_compiler::optimizer {

void LoopPlanner::run() {
  for (auto nst : nest)
    plan(nst);
}

void LoopPlanner::plan(LoopNest &) {}

} // namespace rust_compiler::optimizer
