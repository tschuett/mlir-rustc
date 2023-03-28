#include "Analysis/ScalarEvolution.h"

#include "Analysis/Loops.h"

namespace rust_compiler::analysis {

void ScalarEvolution::analyze(std::span<LoopNest> nests) {

  for (auto &nest : nests)
    analyzeLoopNest(&nest);
}

void ScalarEvolution::analyzeLoopNest(LoopNest *nest) {}

} // namespace rust_compiler::analysis
