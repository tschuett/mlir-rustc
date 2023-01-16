#pragma once

#include "TargetInfo/Struct.h"
#include "TargetInfo/TargetInfo.h"
#include "TargetInfo/Types.h"

#include <span>

namespace rust_compiler::target_info {

StructType *getStruct(std::span<Type *> members, TargetInfo *linux);

}
