#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace rust_compiler::optimizer {

#define GEN_PASS_DECL_TEST
#include "Optimizer/Passes.h.inc"

std::unique_ptr<mlir::Pass> createTestPass();

// declarative passes
#define GEN_PASS_REGISTRATION
#include "Optimizer/Passes.h.inc"

} // namespace rust_compiler::optimizer
