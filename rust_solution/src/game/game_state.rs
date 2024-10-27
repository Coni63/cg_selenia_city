use crate::game::{Module, Pod, Tube};

#[derive(Debug)]
pub struct GameState {
    total_resources: i32,
    tubes: Vec<Tube>,
    pods: Vec<Pod>,
    modules: Vec<Module>,
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            total_resources: 0,
            tubes: Vec::new(),
            pods: Vec::new(),
            modules: Vec::new(),
        }
    }

    pub fn set_ressources(&mut self, resources: i32) {
        self.total_resources = resources;
    }

    pub fn add_tube(&mut self, tube: Tube) {
        if self.tubes.contains(&tube) {
            self.tubes.push(tube);
        }
    }

    pub fn add_pod(&mut self, pod: Pod) {
        if self.pods.contains(&pod) {
            self.pods.push(pod);
        }
    }

    pub fn add_module(&mut self, module: Module) {
        if self.modules.contains(&module) {
            self.modules.push(module);
        }
    }
}
