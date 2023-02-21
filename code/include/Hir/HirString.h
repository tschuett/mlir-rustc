#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/TypeSupport.h>
#include <mlir/IR/Types.h>

namespace rust_compiler::hir {

struct StringTypeStorage : public mlir::TypeStorage {

  using KeyTy = std::pair<uint32_t, llvm::ArrayRef<mlir::Type>>;

  StringTypeStorage(uint32_t active, llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(active, elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static StringTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key.second);

    return new (allocator.allocate<StringTypeStorage>())
        StringTypeStorage(key.first, elementTypes);
  }

  std::pair<uint32_t, llvm::ArrayRef<mlir::Type>> elementTypes;
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, StringTypeStorage> {

  using Base::Base;

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StringType
  get(std::pair<uint32_t, llvm::ArrayRef<mlir::Type>> elementTypes) {
    assert(!elementTypes.second.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.second.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  /// Returns the element types of this struct type.
  std::pair<uint32_t, llvm::ArrayRef<mlir::Type>> getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
  }

  /// Returns the number of element type held by this enum.
  size_t getNumElementTypes() { return getElementTypes().second.size() + 1; }
};

} // namespace rust_compiler::hir
