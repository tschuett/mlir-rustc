
#include "Hir/HirDialect.h"

#include <cstdint>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/SMLoc.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/OpDefinition.h>

#include "Hir/HirTypes.h"

LLVM_ATTRIBUTE_UNUSED static mlir::OptionalParseResult
generatedTypeParser(mlir::AsmParser &parser, llvm::StringRef *mnemonic, mlir::Type &value);
LLVM_ATTRIBUTE_UNUSED static mlir::LogicalResult
generatedTypePrinter(mlir::Type def, mlir::AsmPrinter &printer);

using namespace rust_compiler::hir;

#define GET_TYPEDEF_CLASSES
#include "Hir/HirTypes.cpp.inc"


unsigned
StructType::getTypeSizeInBits(const ::mlir::DataLayout &dataLayout,
                              ::mlir::DataLayoutEntryListRef params) const {
  if (!size)
    computeSizeAndAlignment(dataLayout);
  return *size * 8;
}

unsigned
StructType::getABIAlignment(const ::mlir::DataLayout &dataLayout,
                            ::mlir::DataLayoutEntryListRef params) const {
  if (!align)
    computeSizeAndAlignment(dataLayout);
  return *align;
}

unsigned
StructType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  llvm_unreachable("NYI");
}

bool StructType::isPadded(const ::mlir::DataLayout &dataLayout) const {
  if (!padded)
    computeSizeAndAlignment(dataLayout);
  return *padded;
}

void StructType::computeSizeAndAlignment(
    const ::mlir::DataLayout &dataLayout) const {
  assert(!isOpaque() && "Cannot get layout of opaque structs");
  // Do not recompute.
  if (size || align || padded)
    return;

  // This is a similar algorithm to LLVM's StructLayout.
  unsigned structSize = 0;
  llvm::Align structAlignment{1};
  [[maybe_unused]] bool isPadded = false;
  unsigned numElements = getNumElements();
  auto members = getMembers();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = numElements; i != e; ++i) {
    auto ty = members[i];

    // This matches LLVM since it uses the ABI instead of preferred alignment.
    const llvm::Align tyAlign =
        llvm::Align(getPacked() ? 1 : dataLayout.getTypeABIAlignment(ty));

    // Add padding if necessary to align the data element properly.
    if (!llvm::isAligned(tyAlign, structSize)) {
      isPadded = true;
      structSize = llvm::alignTo(structSize, tyAlign);
    }

    // Keep track of maximum alignment constraint.
    structAlignment = std::max(tyAlign, structAlignment);

    // FIXME: track struct size up to each element.
    // getMemberOffsets()[i] = structSize;

    // Consume space for this data item
    structSize += dataLayout.getTypeSize(ty);
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!llvm::isAligned(structAlignment, structSize)) {
    isPadded = true;
    structSize = llvm::alignTo(structSize, structAlignment);
  }

  size = structSize;
  align = structAlignment.value();
  padded = isPadded;
}

void HirDialect::registerTypes() {
  addTypes <
#define GET_TYPEDEF_LIST
#include "Hir/HirTypes.cpp.inc"
    >();
}



