#include "Mir/MirOps.h"

#include "Mir/MirAttr.h"
#include "Mir/MirDialect.h"
#include "Mir/MirInterfaces.h"
#include "Mir/MirTypes.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

#define GET_OP_CLASSES
#include "Mir/MirOps.cpp.inc"

namespace rust_compiler::Mir {

//===----------------------------------------------------------------------===//
// VTableOp
//===----------------------------------------------------------------------===//

mlir::ParseResult VTableOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  // Parse the name as a symbol reference attribute.
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return mlir::failure();

  if (!failed(parser.parseOptionalKeyword(getExtendsKeyword()))) {
    mlir::StringAttr parent;
    if (parser.parseLParen() ||
        parser.parseAttribute(parent, getParentAttrNameStr(),
                              result.attributes) ||
        parser.parseRParen())
      return mlir::failure();
  }

  // Parse the optional table body.
  mlir::Region *body = result.addRegion();
  mlir::OptionalParseResult parseResult = parser.parseOptionalRegion(*body);
  if (parseResult.has_value() && failed(*parseResult))
    return mlir::failure();

  VTableOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return mlir::success();
}

void VTableOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  if (getParent())
    p << ' ' << getExtendsKeyword() << '('
      << (*this)->getAttr(getParentAttrNameStr()) << ')';

  mlir::Region &body = getOperation()->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/false);
  }
}

mlir::LogicalResult VTableOp::verify() {
  if (getRegion().empty())
    return mlir::success();
  for (auto &op : getBlock())
    if (!mlir::isa<VTEntryOp, MirEndOp>(op))
      return op.emitOpError("vtable must contain vt_entry");
  return mlir::success();
}

void VTableOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                     llvm::StringRef name, mlir::Type type,
                     llvm::StringRef parent,
                     llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  result.addRegion();
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  if (!parent.empty())
    result.addAttribute(getParentAttrNameStr(), builder.getStringAttr(parent));
  // result.addAttribute(getSymbolAttrNameStr(),
  //                     mlir::SymbolRefAttr::get(builder.getContext(), name));
  result.addAttributes(attrs);
}

//===----------------------------------------------------------------------===//
// DTEntryOp
//===----------------------------------------------------------------------===//

mlir::ParseResult VTEntryOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  llvm::StringRef methodName;
  // allow `methodName` or `"methodName"`
  if (failed(parser.parseOptionalKeyword(&methodName))) {
    mlir::StringAttr methodAttr;
    if (parser.parseAttribute(methodAttr, VTEntryOp::getMethodAttrNameStr(),
                              result.attributes))
      return mlir::failure();
  } else {
    result.addAttribute(VTEntryOp::getMethodAttrNameStr(),
                        parser.getBuilder().getStringAttr(methodName));
  }
  mlir::SymbolRefAttr calleeAttr;
  if (parser.parseComma() ||
      parser.parseAttribute(calleeAttr, VTEntryOp::getProcAttrNameStr(),
                            result.attributes))
    return mlir::failure();
  return mlir::success();
}

void VTEntryOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getMethodAttr() << ", " << getProcAttr();
}

} // namespace rust_compiler::Mir
