#include "Hir/HirDialect.h"



void HirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hir/HirOps.cpp.inc"
      >();
  //  addInterfaces<MirInlinerInterface>();
  //addTypes<StructType>();
}
