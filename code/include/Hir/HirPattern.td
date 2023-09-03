#ifndef HirPattern
#define HirPattern

include "mlir/IR/AttrTypeBase.td"
include "HirDialect.td"

class Hir_PatternBase<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Hir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//def Hir_PatternNoTopAlt : Hir_PatternBase<"PatternNoTopAlt", "PatternNoTopAlt"> {
//  let summary = "Rust PatternNoTopAl.";
//
//  let description = [{
//    The type describes a PatternNoTopAlt.
//  }];
//
//}

def Hir_PatternType : Hir_PatternBase<"Pattern", "pattern"> {
  let summary = "Rust Pattern.";

  let description = [{
    The type describes a Pattern, i.e. an Or of PatternNoTopAlts.
  }];

  let parameters = (ins
    ArrayRefParameter<"AnyPatternNoTopAlt", "notopss">:$notopss
  );

  let assemblyFormat = "`<` $notopss `>`";

  let extraClassDeclaration = [{
  }];
}

//def Hir_PatternWithoutRange : Hir_PatternBase<"PatternWithoutRange", "PatternWithoutRange"> {
//  let summary = "Rust PatternWithoutRange.";
//
//  let description = [{
//    The type describes a PatternWithoutRange.
//  }];
//
//  let assemblyFormat = [{
//  }];
//
//  let extraClassDeclaration = [{
//  }];
//}

def Hir_LiteralPattern : Hir_PatternBase<"LiteralPattern", "LiteralPattern"> {
  let summary = "Rust PatternWithoutRange.";

  let description = [{
    The type describes a literal pattern.
  }];

  let assemblyFormat = [{
  }];

  let extraClassDeclaration = [{
  }];
}

def Hir_IdentifierPattern : Hir_PatternBase<"IdentifierPattern", "IdentifierPattern"> {
  let summary = "Rust PatternWithoutRange.";

  let description = [{
    The type describes a identifier pattern.
  }];

  let assemblyFormat = [{
  }];

  let extraClassDeclaration = [{
  }];
}

def Hir_WildcardPattern : Hir_PatternBase<"WildcardPattern", "WildcardPattern"> {
  let summary = "Rust PatternWithoutRange.";

  let description = [{
    The type describes a _ pattern.
  }];

  let assemblyFormat = [{
  }];

  let extraClassDeclaration = [{
  }];
}

def Hir_RestPattern : Hir_PatternBase<"RestPattern", "RestPattern"> {
  let summary = "Rust RestPattern.";

  let description = [{
    The type describes a .. pattern.
  }];

  let assemblyFormat = [{
  }];

  let extraClassDeclaration = [{
  }];
}

def Hir_ReferencePattern : Hir_PatternBase<"ReferencePattern", "ReferencePattern"> {
  let summary = "Rust ReferencePattern.";

  let description = [{
    The type describes a &mut pattern.
  }];

  let parameters = (ins
      AnyPatternNoTopAlt:$pat
  );

  let assemblyFormat = "`<` $pat `>`";

  let extraClassDeclaration = [{
  }];
}

def Hir_TupleStructPattern : Hir_PatternBase<"TupleStructPattern", "TupleStructPattern"> {
  let summary = "Rust TupleStructPattern.";

  let description = [{
    The type describes a tuple struct pattern.
  }];

  let parameters = (ins
    ArrayRefParameter<"AnyPattern", "members">:$members,
    "mlir::StringAttr":$path
  );

  let assemblyFormat = "`<` $path ($members)  `>`";

  let extraClassDeclaration = [{
  }];
}

def Hir_GroupedPattern : Hir_PatternBase<"GroupedPattern", "GroupedPattern"> {
  let summary = "Rust GroupedPattern.";

  let description = [{
    The type describes a (pattern).
  }];

  let parameters = (ins
    AnyPattern:$pat
  );

  let assemblyFormat = "`<` ($pat) `>`";

  let extraClassDeclaration = [{
  }];
}

def Hir_SlicePattern : Hir_PatternBase<"SlicePattern", "SlicePattern"> {
  let summary = "Rust SlicePattern.";

  let description = [{
    The type describes a [ Pattern, ..] pattern.
  }];

  let parameters = (ins
    ArrayRefParameter<"AnyPattern", "members">:$members
  );

  let assemblyFormat = "`<` [ $members ] `>`";

  let extraClassDeclaration = [{
  }];
}

def Hir_PathPattern : Hir_PatternBase<"PathPattern", "PathPattern"> {
  let summary = "Rust PathPattern.";

  let description = [{
    The type describes a path pattern.
  }];

  let parameters = (ins
    "mlir::StringAttr":$path
  );

  let assemblyFormat = "`<` $path `>`";

  let extraClassDeclaration = [{
  }];
}

#endif
