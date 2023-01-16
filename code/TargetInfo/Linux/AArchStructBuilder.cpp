#include "AArchStructBuilder.h"

#include "TargetInfo/TargetInfo.h"

#include <llvm/Support/Casting.h>

namespace rust_compiler::target_info {

// 5.9   Composite Types (64-bit)
// 5.3   Composite Types (32-bit)
StructType *getStruct(std::span<Type *> members, TargetInfo *linux) {
  assert(!members.empty());

  size_t currentOffset = 0;
  size_t currentSize = 0;
  std::vector<std::variant<Type *, unsigned>> newMembers;

  // FIXME bitfields  5.3.4 Bit-fields (32) or 5.9.4   Bit-fields subdivision
  // (64)
  for (unsigned i = 0; i < members.size(); ++i) {
    if (BitfieldType *bit = llvm::dyn_cast<BitfieldType>(members[i])) {
      if (i + 1 < members.size()) {       // has successor?
        if (bit->getFreeBits() / 8 > 0) { // there are bytes?
          size_t align = linux->getAlignmentOf(members[i + 1]);
        }
      }
      // FIXME
    } else {
      size_t align = linux->getAlignmentOf(members[i]);
      size_t size = linux->getSizeOf(members[i]);
      size_t swizzle = currentOffset % align;
      if (swizzle == 0) {
        newMembers.push_back(members[i]);
        currentOffset += size;
        currentSize += size;
      } else {
        newMembers.push_back((unsigned)(align - swizzle)); // padding
        newMembers.push_back(members[i]);                  // value
        currentOffset += size;
        currentSize += size;
      }
    }
  }

  return new StructType(newMembers, currentSize,
                        linux->getAlignmentOf(members[0]));
}

} // namespace rust_compiler::target_info
