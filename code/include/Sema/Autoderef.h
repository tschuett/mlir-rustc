#pragma once

namespace rust_compiler::sema {

enum class AdjustmentKind {};

class Adjustment {
  AdjustmentKind kind;

public:
  AdjustmentKind getKind() const { return kind; }
};

} // namespace rust_compiler::sema
