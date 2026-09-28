package com.codingame.bench;

import com.codingame.gameengine.runner.SoloGameRunner;
import com.codingame.gameengine.runner.simulate.GameResult;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Standalone benchmark runner for the Fall Challenge 2024 - Selenia City game.
 *
 * Usage:
 *   java -jar bench.jar "<agent command>" <configDir> <outputFile> [testNumber]
 *
 * Runs the given agent command against every testN.json file found in
 * <configDir> (or just the given test number if provided), and writes a
 * per-test score summary plus the total score to <outputFile>.
 *
 * test1..test12 are the plain test cases; test13..test24 are their
 * corresponding validators (test(N+12) validates testN) and are what
 * actually counts towards the CodinGame ranking score.
 */
public class BenchRunner {

    private static final Pattern TEST_NUMBER = Pattern.compile("test(\\d+)\\.json$");
    private static final boolean VERBOSE = "1".equals(System.getenv("BENCH_VERBOSE"));
    private static final int VALIDATOR_START = 13;

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: java -jar bench.jar \"<agent command>\" <configDir> <outputFile> [testNumber]");
            System.exit(1);
        }

        String agentCommand = args[0];
        File configDir = new File(args[1]);
        File outputFile = new File(args[2]);
        Integer testNumber = args.length >= 4 ? Integer.parseInt(args[3]) : null;

        List<File> testFiles = listTestFiles(configDir, testNumber);
        if (testFiles.isEmpty()) {
            System.err.println("No test files found in " + configDir.getPath()
                    + (testNumber != null ? " for test number " + testNumber : ""));
            System.exit(1);
        }

        long testsTotal = 0;
        long validatorsTotal = 0;
        int failures = 0;

        try (PrintWriter out = new PrintWriter(new FileWriter(outputFile))) {
            for (File testFile : testFiles) {
                long start = System.currentTimeMillis();

                SoloGameRunner runner = new SoloGameRunner();
                runner.setAgent(agentCommand);
                runner.setTestCase(testFile);

                GameResult result = runner.simulate();
                long elapsedMs = System.currentTimeMillis() - start;

                Long score = extractPoints(result.metadata);
                boolean isValidator = testNumberOf(testFile) >= VALIDATOR_START;
                String line;
                if (score == null) {
                    failures++;
                    String cause = result.failCause != null ? result.failCause : "unknown failure";
                    line = String.format("%-14s FAILED (%s)", testFile.getName(), cause);
                } else {
                    if (isValidator) {
                        validatorsTotal += score;
                    } else {
                        testsTotal += score;
                    }
                    String kind = isValidator ? "validator" : "test     ";
                    line = String.format("%-14s %10d pts  (%5d ms)  [%s]", testFile.getName(), score, elapsedMs, kind);
                }

                System.out.println(line);
                out.println(line);

                if (VERBOSE) {
                    System.err.println("[" + testFile.getName() + "][metadata] " + result.metadata);
                    System.err.println("[" + testFile.getName() + "][scores] " + result.scores);
                    if (result.errors != null) {
                        for (var e : result.errors.entrySet()) {
                            for (String err : e.getValue()) {
                                System.err.println("[" + testFile.getName() + "][" + e.getKey() + " err] " + err);
                            }
                        }
                    }
                    if (result.summaries != null) {
                        for (String s : result.summaries) {
                            System.err.println("[" + testFile.getName() + "][summary] " + s);
                        }
                    }
                }
            }

            long grandTotal = testsTotal + validatorsTotal;
            String testsLine = String.format("TESTS TOTAL (test1-12):      %d pts", testsTotal);
            String validatorsLine = String.format("VALIDATORS TOTAL (test13-24, ranking score): %d pts", validatorsTotal);
            String summary = String.format("TOTAL: %d pts over %d test(s), %d failure(s)",
                    grandTotal, testFiles.size(), failures);

            System.out.println(testsLine);
            System.out.println(validatorsLine);
            System.out.println(summary);
            out.println(testsLine);
            out.println(validatorsLine);
            out.println(summary);
        }

        if (failures > 0) {
            System.exit(1);
        }
    }

    private static List<File> listTestFiles(File configDir, Integer testNumber) {
        File[] files = configDir.listFiles((dir, name) -> TEST_NUMBER.matcher(name).find());
        List<File> result = new ArrayList<>();
        if (files == null) {
            return result;
        }

        for (File f : files) {
            if (testNumber == null || testNumberOf(f) == testNumber) {
                result.add(f);
            }
        }

        result.sort(Comparator.comparingInt(BenchRunner::testNumberOf));
        return result;
    }

    private static int testNumberOf(File f) {
        Matcher m = TEST_NUMBER.matcher(f.getName());
        return m.find() ? Integer.parseInt(m.group(1)) : Integer.MAX_VALUE;
    }

    private static final Pattern POINTS_FIELD = Pattern.compile("\"points\"\\s*:\\s*\"?(-?[0-9.]+)\"?");

    private static Long extractPoints(String metadataJson) {
        if (metadataJson == null) {
            return null;
        }
        Matcher m = POINTS_FIELD.matcher(metadataJson);
        return m.find() ? (long) Double.parseDouble(m.group(1)) : null;
    }
}
