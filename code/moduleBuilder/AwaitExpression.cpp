#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"

using namespace mlir;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitAwaitExpression(
    std::shared_ptr<ast::AwaitExpression> await) {

  mlir::Value body = emitExpression(await->getBody());

  return builder.create<Mir::AwaitOp>(getLocation(await->getLocation()),
                                      TypeRange(builder.getI64Type()), body);
}

} // namespace rust_compiler


// FIXME Type
