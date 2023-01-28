#pragma once

namespace rust_compiler::analysis::attributor {

enum class DepClass {
  REQUIRED, ///< The target cannot be valid if the source is not.
  OPTIONAL, ///< The target may be valid if the source is not.
  NONE,     ///< Do not track a dependence between source and target.
};

}
