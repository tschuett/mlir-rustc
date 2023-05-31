#pragma once

// #include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/ExtensibleDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

namespace rust_compiler::hir {

bool isScalarObject(mlir::Type);

}

#include "HirDialect.h.inc"
