#include "Mir/MirDialect.h"

#include "Mir/MirOps.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/WithColor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/InliningUtils.h>
#include <optional>

#define DEBUG_TYPE "MirDialect"

using namespace mlir;
using namespace rust_compiler::Mir;

#include "Mir/MirDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Mir/MirOps.cpp.inc"

struct MirInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Returns true if the given operation 'callable', that implements the
  /// 'CallableOpInterface', can be inlined into the position given call
  /// operation 'call', that is registered to the current dialect and implements
  /// the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
  /// given 'callable' is set to be cloned during the inlining process, or false
  /// if the region is set to be moved in-place(i.e. no duplicates would be
  /// created).
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    CallableOpInterface callableI =
        dyn_cast<mlir::CallableOpInterface>(callable);
    if (!callableI)
      return false;

    CallOpInterface callI =
        dyn_cast<mlir::CallOpInterface>(call);
    if (!callI)
      return false;

    return false;
  }

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'wouldBeCloned' is set to true if the given 'src' region is set to be
  /// cloned during the inlining process, or false if the region is set to be
  /// moved in-place(i.e. no duplicates would be created). 'valueMapping'
  /// contains any remapped values from within the 'src' region. This can be
  /// used to examine what values will replace entry arguments into the 'src'
  /// region for example.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const override {
    return false;
  }

  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToReplace) const override {
    // Only "Mir.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToReplace[it.index()].replaceAllUsesWith(it.value());
  }

//  Operation *materializeCallConversion(OpBuilder &builder, Value input,
//                                       Type resultType,
//                                       Location conversionLoc) const override {
//    return builder.create<CastOp>(conversionLoc, resultType, input);
//  }
};

void MirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mir/MirOps.cpp.inc"
      >();
  addInterfaces<MirInlinerInterface>();
}

// #include "Mir/MirOps.cpp.inc"

namespace rust_compiler::Mir {} // namespace rust_compiler::Mir
