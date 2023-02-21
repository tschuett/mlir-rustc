#include "CodeGen/Passes.h"
#include "Optimizer/Passes.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

using namespace mlir;

int main(int argc, char **argv) {
  rust_compiler::optimizer::registerOptimizerPasses();
  rust_compiler::codegen::registerCodeGenPasses();
  DialectRegistry registry;
  return failed(MlirOptMain(argc, argv, "rustc modular optimizer driver\n",
                            registry, /*preloadDialectsInContext=*/false));
}
