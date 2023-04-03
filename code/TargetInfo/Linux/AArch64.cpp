#include "AArch64.h"

#include "AArch64TypeBuilder.h"
#include "TargetInfo/Actions.h"
#include "TargetInfo/Struct.h"
#include "TargetInfo/Types.h"

#include <llvm/Support/Casting.h>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <vector>

const uint8_t POINTER_SIZE = 8;

using namespace llvm;
using namespace std;

// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst 2022Q1

// FIXME: 5.9.4   Bit-fields subdivision

namespace rust_compiler::target_info {

bool AArch64Linux::isBigEndian() { return false; }

bool AArch64Linux::isCharSigned() { return false; }

size_t AArch64Linux::getSizeOf(const Type *type) {

  if (auto builtin = llvm::dyn_cast<BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case BuiltinKind::Byte:
      return 1;
    case BuiltinKind::Short:
      return 2;
    case BuiltinKind::Word:
      return 4;
    case BuiltinKind::DoubleWord:
      return 8;
    case BuiltinKind::QuadWord:
      return 16;
    case BuiltinKind::Half:
      return 2;
    case BuiltinKind::Single:
      return 4;
    case BuiltinKind::Double:
      return 8;
    case BuiltinKind::QuadFloat:
      return 16;
    case BuiltinKind::SingleDecimal:
      return 4;
    case BuiltinKind::DoubleDecimal:
      return 8;
    case BuiltinKind::QuadDecimal:
      return 16;
    case BuiltinKind::FloatWth80Bits: {
      assert(false);
      return std::numeric_limits<size_t>::max();
    }
    }
  } else if (auto pointer = dyn_cast<PointerType>(type)) {
    return POINTER_SIZE;
  } else if (auto vector = dyn_cast<VectorType>(type)) {
    size_t bits = vector->getBits();
    assert((bits == 64) || (bits == 128));
    return bits / 8;
  } else if (auto aggre = dyn_cast<StructType>(type)) {
    // 5.9.1 Aggregates
    return aggre->getSizeInBytes();
  } else if (auto unio = dyn_cast<UnionType>(type)) {
    // 5.9.2 Union
    return unio->getSizeInBytes();
  } else if (auto arra = dyn_cast<ArrayType>(type)) {
    // 5.9.3 Array
    Type *baseType = arra->getBaseType();
    return getSizeOf(baseType) * arra->getNumberOfElements();
  } else {
    assert(false);
  }
}

size_t AArch64Linux::getAlignmentOf(const Type *type) {
  if (auto builtin = dyn_cast<BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case BuiltinKind::Byte:
      return 1;
    case BuiltinKind::Short:
      return 2;
    case BuiltinKind::Word:
      return 4;
    case BuiltinKind::DoubleWord:
      return 8;
    case BuiltinKind::QuadWord:
      return 16;
    case BuiltinKind::Half:
      return 2;
    case BuiltinKind::Single:
      return 4;
    case BuiltinKind::Double:
      return 8;
    case BuiltinKind::QuadFloat:
      return 16;
    case BuiltinKind::SingleDecimal:
      return 4;
    case BuiltinKind::DoubleDecimal:
      return 8;
    case BuiltinKind::QuadDecimal:
      return 16;
    case BuiltinKind::FloatWth80Bits: {
      assert(false);
      return std::numeric_limits<size_t>::max();
    }
    }
  } else if (auto pointer = llvm::dyn_cast<PointerType>(type)) {
    return POINTER_SIZE;
  } else if (auto vector = llvm::dyn_cast<VectorType>(type)) {
    size_t bits = vector->getBits();
    assert((bits == 64) || (bits == 128));
    return bits / 8;
  } else if (auto aggre = llvm::dyn_cast<StructType>(type)) {
    // 5.9.1 Aggregates
    return aggre->getAlignmentInBytes();
  } else if (auto uni = llvm::dyn_cast<UnionType>(type)) {
    // 5.9.2 Union
    return uni->getAlignmentInBytes();
  } else if (auto arra = llvm::dyn_cast<ArrayType>(type)) {
    // 5.9.3 Array
    Type *baseType = arra->getBaseType();
    return getAlignmentOf(baseType);
  } else {
    assert(false);
  }
}

CallWithLayoutAndCode AArch64Linux::getCall(/*const FunctionType *signature,*/
                                            span<Type *> arguments) {

  CallWithLayoutAndCode result;

  // Stage B – Pre-padding and extension of arguments
  unsigned idx = 0;
  for (Type *arg : arguments) {
    // B.1
    if (isPureScalabeType(arg)) {
      // do nothing
    }

    // B.2
    if (isDynamicallySizedType(arg)) {
      result.addAction(idx, new CopyToMemoryAction(arg));
    }

    // B.3
    if (isHomogeneousFloatingPointAggregate(arg) ||
        isHomogeneousShortVectorAggregate(arg)) {
      // do nothing
    }
    // B.4
    if (auto struc = llvm::dyn_cast<StructType>(arg)) {
      if (getSizeOf(struc) > 16) {
        result.addAction(idx, new CopyToMemoryAction(arg));
      }
      // B.5.
      result.addAction(idx, new SizeRoundUpAction(arg, 8));
    }
    // B.6.
    // FIXME

    ++idx;
  }

  // Stage C – Assignment of arguments to registers and stack
  idx = 0;
  for (Type *arg : arguments) {
    // C.1
    if (isFloatOrShortVector(arg)) {
      // do nothing
    }
    // C.2
    // skipped
    // C.3
    if (isHomogeneousFloatingPointAggregate(arg) ||
        isHomogeneousShortVectorAggregate(arg)) {
      result.addAction(idx, new SizeRoundUpAction(arg, 8));
    }
    // C.4
    // skipped
    // C.5
    if (isHalfOrSingle(arg)) {
      result.addAction(idx, new SetSizeWithUnspecifiedUpper(arg, 8));
    }
    // C.6
    if (isHomogeneousFloatingPointAggregate(arg) ||
        isHomogeneousShortVectorAggregate(arg) || isFloatOrShortVector(arg)) {
      result.addAction(idx, new CopyToMemoryAction(arg));
    }
    // C.7
    // skipped
    // C.8
    // skipped
    // C.9
    // skipeed
    // C.10
    // skipped
    // C.11
    // skipped
    // C.12
    // skipped
    // C.13
    // skipped
    // C.14
    // skipped
    // C.15
    // skipped
    // C.16
    // skipped
    // C.17
    // skipped

    // 6.5   Result return

    ++idx;
  }

  return result;
}

size_t AArch64Linux::getMaxAlignment(std::span<Type *> members) {
  size_t maxAlign = 0;
  for (Type *mem : members) {
    size_t align = getAlignmentOf(mem);
    maxAlign = std::max(maxAlign, align);
  }
  return maxAlign;
};

// 5.9.5   Homogeneous Aggregates
bool AArch64Linux::isHomogeneousAggregate(const Type *type) {
  if (auto struc = dyn_cast<StructType>(type)) {
    vector<Type *> members = struc->getMembers();
    if (members.empty())
      return true;

    Type::TypeKind kind = members[0]->getKind();
    for (Type *member : members) {
      if (member->getKind() != kind)
        return false;
    }
    return true;
  } else {
    // no struct
    return false;
  }
}

// 5.9.5.1   Homogeneous Floating-point Aggregates (HFA)
bool AArch64Linux::isHomogeneousFloatingPointAggregate(const Type *type) {
  if (not isHomogeneousAggregate(type))
    return false;
  if (auto struc = dyn_cast<StructType>(type)) {
    vector<Type *> members = struc->getMembers();
    if (members.size() > 4)
      return false;
    if (auto mem = dyn_cast<BuiltinType>(members[0])) {
      return mem->isFloatAAPCS();
    } else {
      return false;
    }
  } else {
    return false;
  }
}

// 5.9.5.2   Homogeneous Short-Vector Aggregates (HVA)
bool AArch64Linux::isHomogeneousShortVectorAggregate(const Type *type) {
  if (not isHomogeneousAggregate(type))
    return false;
  if (auto struc = dyn_cast<StructType>(type)) {
    vector<Type *> members = struc->getMembers();
    if (members.size() > 4)
      return false;
    if (auto mem = dyn_cast<VectorType>(members[0])) {
      [[maybe_unused]]size_t bits = mem->getBits();
      // a short vector
      return isShortVector(mem);
    } else {
      return false;
    }
  }
  return false;
}

// 5.10 Pure Scalable Types(PSTs)
bool AArch64Linux::isPureScalabeType(const Type *type) {
  // scalable vector
  if (auto scal = dyn_cast<ScalableVectorType>(type))
    return true;

  // scalable predicate vector
  if (auto scalPred = dyn_cast<ScalablePredicateType>(type))
    return true;

  // array
  if (auto arra = dyn_cast<ArrayType>(type)) {
    if (arra->getNumberOfElements() == 0)
      return false;
    return isPureScalabeType(arra->getBaseType());
  }

  // aggregate
  if (auto struc = dyn_cast<StructType>(type)) {
    if (not isHomogeneousAggregate(struc))
      return false;

    vector<Type *> members = struc->getMembers();
    return isPureScalabeType(members[0]);
  }

  return false;
}

bool AArch64Linux::isShortVector(const Type *type) {
  if (auto vec = dyn_cast<VectorType>(type)) {
    return (vec->getBits() == 64) || (vec->getBits() == 128);
  }
  return false;
}

bool AArch64Linux::isFloatOrShortVector(const Type *type) {
  if (isShortVector(type))
    return true;

  if (auto builtin = dyn_cast<BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case BuiltinKind::Half:
      return true;
    case BuiltinKind::Single:
      return true;
    case BuiltinKind::Double:
      return true;
    case BuiltinKind::QuadFloat:
      return true;
    default:
      return false;
    }
  }

  return false;
}

bool AArch64Linux::isHalfOrSingle(const Type *type) {
  if (auto builtin = dyn_cast<BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case BuiltinKind::Half:
      return true;
    case BuiltinKind::Single:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool AArch64Linux::isFloat(const Type *type) {
  if (auto builtin = dyn_cast<BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case BuiltinKind::Half:
      return true;
    case BuiltinKind::Single:
      return true;
    case BuiltinKind::Double:
      return true;
    case BuiltinKind::QuadFloat:
      return true;
    default:
      return false;
    }
  }

  return false;
}

TypeBuilder *AArch64Linux::getTypeBuilder() {
  return new AArch64LinuxTypeBuilder(this);
}

unsigned AArch64Linux::getNrOfBitsInLargestLockFreeInteger() const {
  return 128;
}

bool AArch64Linux::isDynamicallySizedType(Type *type) {
  if (auto aggre = llvm::dyn_cast<StructType>(type)) {
    for (auto membr : aggre->getMembers()) {
      if (llvm::isa<ScalablePredicateType>(membr)) {
        return true;
      }
      if (llvm::isa<ScalableVectorType>(membr)) {
        return true;
      }
      if (auto aggre = llvm::dyn_cast<StructType>(membr)) {
        if (isDynamicallySizedType(aggre)) {
          return true;
        }
      }
    }
    return false;
  } else {
    return false;
  }
}

} // namespace rust_compiler::target_info
