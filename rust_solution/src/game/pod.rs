use crate::utils::macros::parse_input;

#[derive(Debug)]
pub struct Pod {
    id: i32,
    num_nodes: i32,
    nodes: Vec<i32>,
}

impl Pod {
    pub fn new(id: i32, num_nodes: i32, nodes: Vec<i32>) -> Self {
        Pod {
            id,
            num_nodes,
            nodes,
        }
    }
}

impl From<String> for Pod {
    fn from(s: String) -> Self {
        let mut split = s.split(" ").map(|s| parse_input!(s, i32));
        let id = split.next().unwrap();
        let num_nodes = split.next().unwrap();
        let nodes = split.collect();
        Self::new(id, num_nodes, nodes)
    }
}

impl PartialEq for Pod {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
