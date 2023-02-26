#pragma once

#include "AST/InnerAttribute.h"
#include "AST/OuterAttribute.h"

#include <llvm/ADT/StringRef.h>

namespace rust_compiler::sema {

enum BuiltinKind {
  Cfg,
  CfgAttr,
  Test,
  Ignore,
  ShouldPanic,
  Derive,
  AutomaticallDerived,
  MacroExport,
  MacroUse,
  ProcMacro,
  ProcMacroDerive,
  ProcMacroAttribute,
  Allow,
  Warn,
  Deny,
  Forbid,
  Deprecated,
  MusUse,
  Link,
  LinkName,
  LinkOrdinal,
  NoLink,
  Repr,
  CrateType,
  NoMain,
  ExportName,
  LinkSection,
  NoMangle,
  Used,
  CrateName,
  Inline,
  Cold,
  NoBuiltins,
  TargetFeature,
  TrackCaller,
  InstructionSet,
  Doc,
  Comments,
  NoStd,
  NoImplicitPrelude,
  Path,
  RecursionLimit,
  TypeLengthLimit,
  PanicHandler,
  GlobalAllocator,
  WindowsSubsystem,
  Feature,
  NonExhaustive
};

class AttributeAnalyzer {

public:
  AttributeAnalyzer() = default;

  void analyzeOuterAttribute(const ast::OuterAttribute &attr);
  void analyzeInnerAttribute(const ast::InnerAttribute &attr);

private:
  void checkBuiltin(llvm::StringRef path);
  void analyzeRepr(const ast::AttrInput&);
  void analyzeDerive(const ast::AttrInput&);
};

} // namespace rust_compiler::sema
