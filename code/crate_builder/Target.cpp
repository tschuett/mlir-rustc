#include "CrateBuilder/Target.h"

namespace rust_compiler::crate_builder {

  Target::Target(llvm::TargetMachine *_tm) { tm = _tm; }

} // namespace rust_compiler::crate_builder
