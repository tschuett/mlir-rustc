struct Container {}

impl Container {
    fn empty(self) -> bool {
        return true;
    }

    fn new() -> Self {
        return Container {};
    }
}

fn main() {
    let container: Container = Container::new();
    container.empty();
}
