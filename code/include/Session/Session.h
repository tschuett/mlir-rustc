#pragma once

#include "Basic/Ids.h"

namespace rust_compiler::tyctx {
class TyCtx;
}

namespace rust_compiler::session {

class Session {
  basic::CrateNum currenteCrate;
  tyctx::TyCtx *ctx;

public:
  Session(basic::CrateNum currenteCrate, tyctx::TyCtx *ctx)
      : currenteCrate(currenteCrate), ctx(ctx) {}

  void setTypeContext(tyctx::TyCtx *_ctx) { ctx = _ctx; }
  basic::CrateNum getCurrentCrateNum() const;
  tyctx::TyCtx *getTypeContext() const;
};

extern Session *session;

} // namespace rust_compiler::session
