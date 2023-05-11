#pragma once

#include "Basic/Ids.h"
#include "TyCtx/TyCtx.h"

namespace rust_compiler::tyctx {

class TraitQueryGuard {
public:
  TraitQueryGuard(basic::NodeId id, TyCtx *ctx) : id(id), ctx(ctx) {
    ctx->insertTraitQuery(id);
  }

  ~TraitQueryGuard() { ctx->traitQueryCompleted(id); }

private:
  basic::NodeId id;
  TyCtx *ctx;
};

} // namespace rust_compiler::tyctx
