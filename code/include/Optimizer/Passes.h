#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace rust_compiler::optimizer {

#define GEN_PASS_DECL_TEST
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_ATTRIBUTERLITE
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_REWRITEPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_LOWERAWAITPASS
#include "Optimizer/Passes.h.inc"

std::unique_ptr<mlir::Pass> createTestPass();
std::unique_ptr<mlir::Pass> createAttributerPass();
std::unique_ptr<mlir::Pass> createRewriterPass();
  std::unique_ptr<mlir::Pass> createLowerAwaitPass();

// declarative passes
#define GEN_PASS_REGISTRATION
#include "Optimizer/Passes.h.inc"

} // namespace rust_compiler::optimizer
