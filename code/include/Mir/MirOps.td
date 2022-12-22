include "MirTypes.td"

class Mir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mir_Dialect, mnemonic, traits>;

def Mir_AddOp : Mir_Op<"Add", []> {
  let summary = "Performs addition on the input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];
}
