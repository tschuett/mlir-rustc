include "llvm/Option/OptParser.td"


class F<string name, string help> : Flag<["-"], name>, HelpText<help>;
class Arg<string name, string help> : Separate<["-"], name>, HelpText<help>;

def help : Flag<["--"], "help">, HelpText<"Display this help">;
def version : Flag<["--"], "version">, HelpText<"Display the version">;

