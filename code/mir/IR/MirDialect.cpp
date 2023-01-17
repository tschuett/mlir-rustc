#include "Mir/MirDialect.h"

#include "Mir/MirAttr.h"
#include "Mir/MirDialect.h"
#include "Mir/MirInterfaces.h"
#include "Mir/MirOps.h"
#include "Mir/MirTypes.h"

#include <llvm/Support/Debug.h>
#include <llvm/Support/WithColor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/InliningUtils.h>
#include <optional>

#define DEBUG_TYPE "MirDialect"

using namespace mlir;
using namespace llvm;
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
        llvm::dyn_cast<mlir::CallableOpInterface>(callable);
    if (!callableI)
      return false;

    CallOpInterface callI = llvm::dyn_cast<mlir::CallOpInterface>(call);
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
    auto returnOp = llvm::cast<mlir::func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToReplace.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToReplace[it.index()].replaceAllUsesWith(it.value());
  }

  //  Operation *materializeCallConversion(OpBuilder &builder, Value input,
  //                                       Type resultType,
  //                                       Location conversionLoc) const
  //                                       override {
  //    return builder.create<CastOp>(conversionLoc, resultType, input);
  //  }
};

void MirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mir/MirOps.cpp.inc"
      >();
  addInterfaces<MirInlinerInterface>();
  addTypes<StructType>();
}

namespace rust_compiler::Mir {

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
  assert(!elementTypes.empty() && "expected at least 1 element type");

  // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
  // of this type. The first parameter is the context to unique in. The
  // parameters after the context are forwarded to the storage instance.
  mlir::MLIRContext *ctx = elementTypes.front().getContext();
  return Base::get(ctx, elementTypes);
}

/// Returns the element types of this struct type.
llvm::ArrayRef<mlir::Type> StructType::getElementTypes() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementTypes;
}

/// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
  if (type.isa<mlir::TensorType>()) {
    // Check that the value is an elements attribute.
    auto attrValue = opaqueValue.dyn_cast<mlir::DenseFPElementsAttr>();
    if (!attrValue)
      return op->emitError("constant of TensorType must be initialized by "
                           "a DenseFPElementsAttr, got ")
             << opaqueValue;

    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = type.dyn_cast<mlir::RankedTensorType>();
    if (!resultType)
      return success();

    // Check that the rank of the attribute type matches the rank of the
    // constant result type.
    auto attrType = attrValue.getType().cast<mlir::TensorType>();
    if (attrType.getRank() != resultType.getRank()) {
      return op->emitOpError("return type must match the one of the attached "
                             "value attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        return op->emitOpError(
                   "return type shape mismatches its attribute at dimension ")
               << dim << ": " << attrType.getShape()[dim]
               << " != " << resultType.getShape()[dim];
      }
    }
    return mlir::success();
  }
  auto resultType = type.cast<StructType>();
  llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

  // Verify that the initializer is an Array.
  auto attrValue = opaqueValue.dyn_cast<ArrayAttr>();
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitError("constant of StructType must be initialized by an "
                         "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid.
  llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult StructConstantOp::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}

mlir::LogicalResult StructAccessOp::verify() {
  StructType structTy = getInput().getType().cast<StructType>();
  size_t indexValue = getIndex();
  if (indexValue >= structTy.getNumElementTypes())
    return emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[indexValue])
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  return mlir::success();
}

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr = adaptor.getInput().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = getIndex();
  return structAttr[elementIndex];
}

} // namespace rust_compiler::Mir
