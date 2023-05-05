#pragma once

#include <vector>

namespace rust_compiler::tyctx::TyTy {
class BaseType;
}

namespace rust_compiler::sema {

using namespace rust_compiler::tyctx;

enum class AdjustmentKind {
  ImmutableReference,
  MutableReference,
  Deref,
  DerefMut,
  Indirection,
  Unsize,

  Error
};

class Adjustment {
  AdjustmentKind kind;

public:
  AdjustmentKind getKind() const { return kind; }
};

class Adjuster {
public:
  Adjuster(const TyTy::BaseType *base) : base(base) {}

  TyTy::BaseType *adjustType(const std::vector<Adjustment> &adjustments);

  const TyTy::BaseType *getType() const { return base; }

private:
  const TyTy::BaseType *base;
};

} // namespace rust_compiler::sema
