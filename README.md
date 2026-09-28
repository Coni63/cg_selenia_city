# Setup

### Game source

The CodinGame game source lives in `FallChallenge2024-SeleniaCity/` (cloned from
https://github.com/0x6E0FF/FallChallenge2024-SeleniaCity). It includes a custom
`com.codingame.bench.BenchRunner` (`src/main/java/com/codingame/bench/BenchRunner.java`)
and a `maven-shade-plugin` config in `pom.xml` that together produce a standalone
benchmark jar — no CodinGame IDE/export needed.

Build it with:

```sh
cd FallChallenge2024-SeleniaCity
mvn -q -DskipTests package
```

This produces `FallChallenge2024-SeleniaCity/target/fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar`,
which runs every `testN.json` in `FallChallenge2024-SeleniaCity/config/` against an
agent command and reports the score (parsed from the referee's `points` metadata) for each.

### Python solution dependencies

```sh
cd python_solution
poetry install --no-root
```

### Running the benchmark

`bench_full.py` (at the repo root) wraps the jar: it builds it if missing, picks the
`python_solution/.venv` interpreter by default, and runs the full test suite.

```sh
python bench_full.py                # full 24-test benchmark
python bench_full.py --test 8       # single test case
python bench_full.py --rebuild      # force a fresh mvn package before running
```

To benchmark a different solver (e.g. a future Rust port), pass `--solution`:

```sh
python bench_full.py --solution rust_solution/target/release/agent.exe
```

Under the hood this is equivalent to:

```sh
java -jar FallChallenge2024-SeleniaCity/target/fall-challenge-2024-moon-city-1.0-SNAPSHOT.jar \
     "<python_solution/.venv python> python_solution/main.py" \
     FallChallenge2024-SeleniaCity/config ref_scores.txt [testNumber]
```

Set `BENCH_VERBOSE=1` to print each test's stderr/summary output for debugging.

### Sources

- https://www.codingame.com/forum/t/fall-challenge-2024-feedback-and-strategies/205205/11

- https://github.com/mourner/delaunator-rs

- https://virtual-atom.com/codingame/fall24/
