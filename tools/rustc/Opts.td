include "llvm/Option/OptParser.td"


multiclass B<string name, string help1, string help2> {
  def NAME: Flag<["--"], name>, HelpText<help1>;
  def no_ # NAME: Flag<["--"], "no-" # name>, HelpText<help2>;
}

multiclass Eq<string name, string help> {
  def NAME : Separate<["--"], name>;
  def NAME #_eq : Joined<["--"], name #"=">,
                  Alias<!cast<Separate>(NAME)>,
                  HelpText<help>;
}

def help : Flag<["--"], "help">, HelpText<"Display this help">;
def version : Flag<["--"], "version">, HelpText<"Display the version">;


def edition_EQ : Joined<["--"], "edition=">,
  HelpText<"Rust edition">;

def path_EQ : Joined<["--"], "path=">,
  HelpText<"Path to Cargo.toml directory">;

def path_EQ : Joined<["--"], "crate-name">,
  HelpText<"The name of the crate to build">;

def path_EQ : Joined<["--"], "out-dir">,
  HelpText<"Directory to write the output in">;
