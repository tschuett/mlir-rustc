#include "Sema/AttributeAnalyzer.h"

#include "AST/AttrInput.h"
#include "AST/Expression.h"

#include <llvm/ADT/StringSwitch.h>

using namespace llvm;
using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void AttributeAnalyzer::analyzeOuterAttribute(const OuterAttribute &attr) {
//  SimplePath path = attr.getPath();
//  if (path.getLength() == 1) {
//  }
}

void AttributeAnalyzer::analyzeInnerAttribute(const InnerAttribute &attr) {
//  SimplePath path = attr.getPath();
//  if (path.getLength() == 1) {
//  }
}

void AttributeAnalyzer::checkBuiltin(llvm::StringRef path) {

//  BuiltinKind builtin = StringSwitch<BuiltinKind>(path)
//                            .Case("cfg", BuiltinKind::Cfg)
//                            .Case("cfg_attr", BuiltinKind::CfgAttr)
//                            .Case("test", BuiltinKind::Test)
//                            .Case("ignore", BuiltinKind::Ignore)
//                            .Case("should_panic", BuiltinKind::ShouldPanic)
//                            .Case("derive", BuiltinKind::Derive);
}

void AttributeAnalyzer::analyzeRepr(const AttrInput &input) {}

void AttributeAnalyzer::analyzeDerive(const AttrInput &) {}

} // namespace rust_compiler::sema
