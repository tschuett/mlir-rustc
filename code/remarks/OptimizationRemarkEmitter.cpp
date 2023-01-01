#include "Remarks/OptimizationRemarkEmitter.h"

#include <llvm/Remarks/Remark.h>

using namespace llvm;

namespace rust_compiler::remarks {

void OptimizationRemarkEmitter::emit(OptimizationRemarkBase &OptDiag) {

  auto result = OptDiag();

  llvm::remarks::Remark remark;
  serializer->emit(remark);
}


} // namespace rust_compiler::remarks
