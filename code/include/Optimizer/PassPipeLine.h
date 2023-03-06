#pragma once

#include <mlir/Pass/PassManager.h>
#include <string_view>

namespace rust_compiler::optimizer {

void createDefaultOptimizerPassPipeline(mlir::PassManager &pm,
                                        std::string_view summaryFile);

}
