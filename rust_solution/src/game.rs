use crate::point::Point;

#[derive(Debug)]
pub struct Game {
    total_ressources: i32,
    num_lines: usize,
    lines: Vec<Tube>,
    num_capsules: usize,
    cqpsules: Vec<Capsule>,
    num_modules: usize,
    modules: Vec<Module>,
}

impl Game {
    pub fn new() -> Self {
        Game {
            total_ressources: 0,
            num_lines: 0,
            lines: Vec::new(),
            num_capsules: 0,
            cqpsules: Vec::new(),
            num_modules: 0,
            modules: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Capsule {
    pub id: usize,
    pub num_nodes: i32,
    pub nodes: Vec<usize>,
}

#[derive(Debug)]
pub struct Tube {
    pub from_node: usize,
    pub to_node: usize,
    pub level: u8,
}

#[derive(Debug)]
pub struct Module {
    pub id: usize,
    pub type_: u8,
    pub position: Point,
    pub nodes: Vec<usize>,
    pub astronaute: [i32; 20],
}
