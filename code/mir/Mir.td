include "mlir/IR/DialectBase.td"

// Here is a simple definition of a dialect.
def MirDialect : Dialect {
  let summary = "A short one line description of my dialect.";
  let description = [{
    My dialect is a very important dialect. This section contains a much more
    detailed description that documents all of the important pieces of information
    to know about the document.
  }];

  /// This is the namespace of the dialect. It is used to encapsulate the sub-components
  /// of the dialect, such as operations ("my_dialect.foo").
  let name = "Mir Dialect";

  /// The C++ namespace that the dialect, and its sub-components, get placed in.
  let cppNamespace = "::mir_dialect";
}
