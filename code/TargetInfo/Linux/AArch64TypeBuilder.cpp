#include "AArch64TypeBuilder.h"

#include "AArchStructBuilder.h"
#include "TargetInfo/Struct.h"
#include "TargetInfo/Types.h"

#include <cstddef>
#include <llvm/Support/Casting.h>

using namespace std;

namespace rust_compiler::target_info {

BuiltinType *AArch64LinuxTypeBuilder::getBuiltin(BuiltinKind bk) {
  assert(bk != BuiltinKind::FloatWth80Bits);

  return new BuiltinType(bk);
}

// 5.9   Composite Types
StructType *AArch64LinuxTypeBuilder::getStruct(std::span<Type *> members) {
  return rust_compiler::target_info::getStruct(members, target);
}

bool AArch64LinuxTypeBuilder::isSupportedVectorLength(size_t bits) const {
  return (bits == 64) || (bits == 128);
}

bool AArch64LinuxTypeBuilder::isSupportedBitFieldType(const Type *type) const {
  return llvm::isa<BuiltinType>(type) || llvm::isa<PointerType>(type) ||
         llvm::isa<VectorType>(type);
}

} // namespace rust_compiler::target_info
