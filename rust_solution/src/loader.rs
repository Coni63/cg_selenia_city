use std::io;

use crate::game::{GameState, Module, Pod, Tube};
use crate::utils::macros::parse_input;

pub fn load(game: &mut GameState) {
    // game loop
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let resources = parse_input!(input_line, i32);
    game.set_ressources(resources);

    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_travel_routes = parse_input!(input_line, i32);
    for _ in 0..num_travel_routes as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(" ").collect::<Vec<_>>();
        let building_id_1 = parse_input!(inputs[0], i32);
        let building_id_2 = parse_input!(inputs[1], i32);
        let capacity = parse_input!(inputs[2], i32);
        let tube = Tube::new(building_id_1, building_id_2, capacity);
        game.add_tube(tube);
    }

    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_pods = parse_input!(input_line, i32);
    for _ in 0..num_pods as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let pod_properties = input_line.trim_matches('\n').to_string();
        let pod = Pod::from(pod_properties);
        game.add_pod(pod);
    }

    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_new_buildings = parse_input!(input_line, i32);
    for _ in 0..num_new_buildings as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let building_properties = input_line.trim_matches('\n').to_string();
        let building = Module::from(building_properties);
        game.add_module(building);
    }

    // Write an action using println!("message...");
    // To debug: eprintln!("Debug message...");
}
