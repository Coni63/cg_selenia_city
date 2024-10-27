use crate::utils::macros::parse_input;

use super::point::Point;

#[derive(Debug)]
pub struct Module {
    id: i32,
    type_: i32,
    position: Point,
    crew: [i32; 20],
}

impl Module {
    pub fn new(id: i32, type_: i32, position: Point, crew: [i32; 20]) -> Self {
        Module {
            id,
            type_,
            position,
            crew,
        }
    }
}

impl From<String> for Module {
    fn from(s: String) -> Self {
        let mut split = s.split(" ").map(|s| parse_input!(s, i32));
        let type_ = split.next().unwrap();
        let id = split.next().unwrap();
        let x = split.next().unwrap() as f64;
        let y = split.next().unwrap() as f64;
        let position = Point { x, y };
        let _ = split.next().unwrap_or(0);
        let mut crew = [0; 20];
        for c in split {
            crew[c as usize] += 1;
        }

        Self::new(id, type_, position, crew)
    }
}

impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
