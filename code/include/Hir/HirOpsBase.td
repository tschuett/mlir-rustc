#ifndef HirOpsBase
#define HirOpsBase

include "mlir/IR/AttrTypeBase.td"

def IsScalarPred
      : CPred<"rust_compiler::hir::isScalarObject($_self)">;
def AnyScalarObject: Type<IsScalarPred, "any scalar-like object">;

def IsPatternPred
    : CPred<"rust_compiler::::hir::isPattern($_self)">;
def AnyPattern: Type<IsPatternPred, "any pattern">;

def IsPatternNoTopAltPred
    : CPred<"rust_compiler::hir::isPatternNoTopAlt($_self)">;
def AnyPatternNoTopAlt: Type<IsPatternNoTopAltPred, "any no top alt pattern">;

def IsPatternWithoutRange
    : CPred<"rust_compiler::hir::isPatternWithoutRange($_self)">;

def IsRangePattern
    : CPred<"rust_compiler::hir::isRangePattern($_self)">;


#endif
