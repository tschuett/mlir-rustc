#include "Lir/LirDialect.h"

#include "Lir/LirAttr.h"
#include "Lir/LirDialect.h"
#include "Lir/LirInterfaces.h"
#include "Lir/LirOps.h"
#include "Lir/LirTypes.h"

#define DEBUG_TYPE "LirDialect"

using namespace mlir;
using namespace llvm;
using namespace rust_compiler::Lir;

#include "Lir/LirDialect.cpp.inc"


void LirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Lir/LirOps.cpp.inc"
      >();
  //  addInterfaces<MirInlinerInterface>();
  //addTypes<StructType>();
}

