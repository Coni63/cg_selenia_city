# Setup

### Build the game

Follow documentation on https://github.com/0x6E0FF/FallChallenge2024-SeleniaCity/tree/main

### Run the game

#### Rust version

On the rust project, build and run the provided agent:

**Debug mode:**

```sh
cargo build
java -jar ../FallChallenge2024-SeleniaCity/target/fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar "target/debug/agent.exe" ../FallChallenge2024-SeleniaCity/config/ ref_scores.txt
```

**Release mode:**

```sh
cargo build --release
java -jar ../FallChallenge2024-SeleniaCity/target/fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar "target/release/agent.exe" ../FallChallenge2024-SeleniaCity/config/ ref_scores.txt
```

**Precompiled version (may be outdated):**

```sh
cargo build --release
java -jar fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar "rust_solution/target/release/agent.exe" config/ ref_scores.txt
```

**Run single game**

```sh
cargo build --release
java -jar fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar "rust_solution/target/release/agent.exe" config/ ref_scores.txt 1
                                                                                                                       # 1 is the game number
```

#### Python version

**Precompiled version (may be outdated):**

If you are using a venv, you may need to change the path of the venv as default python interpreter:

```sh
set PATH=F:\admin\PycharmProjects\CG\Competition\optimisation\cg_selenia_city\python_solution\.venv\Scripts;%PATH%
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ adjust
```

Then you can run it

```sh
java -jar fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar "python python_solution/main.py" config/ ref_scores.txt
```

### Sources

- https://www.codingame.com/forum/t/fall-challenge-2024-feedback-and-strategies/205205/11

- https://github.com/mourner/delaunator-rs
