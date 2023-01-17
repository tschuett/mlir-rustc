#include "Mir/MirOps.h"

#include "Mir/MirAttr.h"
#include "Mir/MirTypes.h"
#include "Mir/MirInterfaces.h"

#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>
#include <mlir/IR/BuiltinAttributes.h>

#define GET_OP_CLASSES
#include "Mir/MirOps.cpp.inc"
