#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace rust_compiler::optimizer {

#define GEN_PASS_DECL_TEST
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_ATTRIBUTER
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_REWRITEPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_LOWERAWAITPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_LOWERERRORPROPAGATIONPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_SUMMARYWRITERPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_GVNPASS
#include "Optimizer/Passes.h.inc"

std::unique_ptr<mlir::Pass> createRewriterPass();

// declarative passes
#define GEN_PASS_REGISTRATION
#include "Optimizer/Passes.h.inc"

} // namespace rust_compiler::optimizer
