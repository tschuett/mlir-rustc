#include "LoopPlan.h"

namespace rust_compiler::optimizer {

LoopPlanner::LoopPlanner(analysis::LoopNest *nest) { this->nest = nest; }

} // namespace rust_compiler::optimizer
