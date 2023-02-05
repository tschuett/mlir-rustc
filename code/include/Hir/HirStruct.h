#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

namespace rust_compiler::hir {

struct StructTypeStorage : public mlir::TypeStorage {

  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  llvm::ArrayRef<mlir::Type> elementTypes;
};

class StructType
    : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {

  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
  }

  /// Returns the number of element type held by this enum.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};

} // namespace rust_compiler::hir
