#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "MirOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "MirOps.h.inc"
