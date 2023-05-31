
#include "Hir/HirDialect.h"

#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/OpDefinition.h>

#include "Hir/HirTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Hir/HirTypes.cpp.inc"
