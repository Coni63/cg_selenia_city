use std::io;

macro_rules! parse_input {
    ($x:expr, $t:ident) => {
        $x.trim().parse::<$t>().unwrap()
    };
}

pub fn load() {
    // game loop
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let resources = parse_input!(input_line, i32);
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_travel_routes = parse_input!(input_line, i32);
    for i in 0..num_travel_routes as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(" ").collect::<Vec<_>>();
        let building_id_1 = parse_input!(inputs[0], i32);
        let building_id_2 = parse_input!(inputs[1], i32);
        let capacity = parse_input!(inputs[2], i32);
    }
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_pods = parse_input!(input_line, i32);
    for i in 0..num_pods as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let pod_properties = input_line.trim_matches('\n').to_string();
    }
    let mut input_line = String::new();
    io::stdin().read_line(&mut input_line).unwrap();
    let num_new_buildings = parse_input!(input_line, i32);
    for i in 0..num_new_buildings as usize {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let building_properties = input_line.trim_matches('\n').to_string();
    }

    // Write an action using println!("message...");
    // To debug: eprintln!("Debug message...");
}
