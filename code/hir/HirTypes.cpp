
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

LLVM_ATTRIBUTE_UNUSED static mlir::OptionalParseResult
generatedTypeParser(mlir::AsmParser &parser, llvm::StringRef *mnemonic, mlir::Type &value);
LLVM_ATTRIBUTE_UNUSED static mlir::LogicalResult
generatedTypePrinter(mlir::Type def, mlir::AsmPrinter &printer);

#define GET_TYPEDEF_CLASSES
#include "Hir/HirTypes.cpp.inc"

#include <llvm/Support/Compiler.h>

using namespace mlir;


