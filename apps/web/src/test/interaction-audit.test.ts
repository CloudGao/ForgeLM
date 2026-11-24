import { describe, expect, it } from "vitest";

const sourceFiles = import.meta.glob<string>(["../**/*.tsx", "!../**/*.test.tsx"], { query: "?raw", import: "default", eager: true });

describe("interaction audit", () => {
  it("does not render active buttons without an interaction", () => {
    const violations = Object.entries(sourceFiles).flatMap(([path, source]) => {
      return Array.from(source.matchAll(/<button\b[^>]*>/g))
        .filter(([tag]) => !/\bonClick=|\btype=["']submit["']|\bdisabled(?:\s|=|>)/.test(tag))
        .map((match) => `${path}:${source.slice(0, match.index).split("\n").length} ${match[0]}`);
    });

    expect(violations).toEqual([]);
  });
});
