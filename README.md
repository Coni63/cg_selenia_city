# Setup

### Build the game

Follow documentation on https://github.com/0x6E0FF/FallChallenge2024-SeleniaCity/tree/main

### Run the game

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
java -jar fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar "target/release/agent.exe" config/ ref_scores.txt
```
