#include "Session/Session.h"

namespace rust_compiler::session {

basic::CrateNum Session::getCurrentCrateNum() const { return currenteCrate; }

tyctx::TyCtx *Session::getTypeContext() const { return ctx; }

Session *session;

} // namespace rust_compiler::session
