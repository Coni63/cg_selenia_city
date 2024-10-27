pub mod macros {
    macro_rules! parse_input {
        ($x:expr, $t:ident) => {
            $x.trim().parse::<$t>().unwrap()
        };
    }

    pub(crate) use parse_input;
}
