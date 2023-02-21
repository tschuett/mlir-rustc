#include "Analysis/Attributer/AANoop.h"

namespace rust_compiler::analysis::attributor {

void AANoop::initialize(Attributor &solver) {}

const std::string AANoop::getAsStr(mlir::AsmState &asmState) const {
  return "false";
}

} // namespace rust_compiler::analysis::attributor
