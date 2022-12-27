#pragma once

#include <mlir/Analysis/AliasAnalysis.h>
#include <optional>

namespace rust_compiler::analysis {

std::optional<mlir::AliasResult>
mayAlias(mlir::AliasAnalysis *alias, mlir::Operation *a, mlir::Operation *b);

}
