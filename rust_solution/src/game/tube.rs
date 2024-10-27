#[derive(Debug, Clone)]
pub struct Tube {
    from_node: i32,
    to_node: i32,
    level: i32,
}

impl Tube {
    pub fn new(from_node: i32, to_node: i32, level: i32) -> Self {
        if from_node > to_node {
            Tube {
                from_node: to_node,
                to_node: from_node,
                level,
            }
        } else {
            Tube {
                from_node,
                to_node,
                level,
            }
        }
    }
}

impl PartialEq for Tube {
    fn eq(&self, other: &Self) -> bool {
        self.from_node == other.from_node && self.to_node == other.to_node
    }
}

impl Eq for Tube {}

impl std::hash::Hash for Tube {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let custom_hash = 1000 * self.from_node + self.to_node;
        // custom_hash.hash(state);
        state.write_u64(custom_hash as u64);
    }
}
