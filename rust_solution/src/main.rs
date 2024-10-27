use game::GameState;

mod delaunator;
mod game;
mod loader;
mod utils;
/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/
fn main() {
    // game loop
    loop {
        let mut game = GameState::new();
        loader::load(&mut game);

        eprintln!("{:?}", game);

        println!("WAIT")
    }
}
