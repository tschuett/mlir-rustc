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

#define GEN_PASS_DECL_DEADCODEELIMINATIONPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_DEADSTOREELIMINATIONPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_SCCPPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_HIRLICMPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_CONVERTHIRTOMIRPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_CONVERTMIRTOLIRPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_CONVERTLIRTOLLVMPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_LOOPPASS
#include "Optimizer/Passes.h.inc"

#define GEN_PASS_DECL_DEADARGUMENTELIMINATIONPASS
#include "Optimizer/Passes.h.inc"

// declarative passes
#define GEN_PASS_REGISTRATION
#include "Optimizer/Passes.h.inc"

} // namespace rust_compiler::optimizer
