#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace rust_compiler::codegen {

#define GEN_PASS_DECL_MIRTOLLVMLOWERING
#include "Optimizer/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLowerMirToLLVMPass();

// declarative passes
#define GEN_PASS_REGISTRATION
#include "CodeGen/Passes.h.inc"

} // namespace rust_compiler::optimizer
