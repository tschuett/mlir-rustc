#pragma once

#include "Mir/MirOpsInterfaces.h.inc"

namespace rust_compiler::Mir {

template <typename T>
using isMirMethod = std::is_base_of<MirMethodOpInterface::Trait<T>, T>;

}
