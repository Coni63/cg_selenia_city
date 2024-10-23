mod delaunator;
mod game;
mod loader;
mod point;

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
fn main() {
    // game loop
    loop {
        let game = loader::load();

        eprintln!("{:?}", game);

        println!("WAIT")
    }
}
