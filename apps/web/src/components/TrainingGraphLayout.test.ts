import { describe, expect, it } from "vitest";
import { layoutTrainingGraph } from "./TrainingGraphLayout";

describe("training graph layout", () => {
  it("keeps variable-height nodes and routed edges apart, including long bypass edges", async () => {
    const nodes = Array.from({ length: 9 }, (_, index) => ({ id: String(index), data: {}, position: { x: 0, y: 0 }, measured: { width: 260, height: 130 + index % 3 * 95 } }));
    const edges = [[0, 3], [1, 3], [2, 3], [2, 4], [3, 5], [3, 6], [4, 6], [5, 7], [6, 7], [3, 7], [7, 8], [0, 8]].map(([source, target], index) => ({ id: `edge-${index}`, source: String(source), target: String(target) }));
    const before = structuredClone({ nodes, edges });
    const layout = await layoutTrainingGraph(nodes, edges);
    const boxes = nodes.map(node => ({ id: node.id, ...layout.nodes[node.id].position, width: node.measured.width, height: layout.nodes[node.id].height }));
    for (const a of boxes) for (const b of boxes) if (a.id !== b.id) {
      expect(a.x + a.width <= b.x || b.x + b.width <= a.x || a.y + a.height <= b.y || b.y + b.height <= a.y).toBe(true);
    }
    for (const edge of edges) {
      const route = layout.edges[edge.id];
      expect(route.points.length).toBeGreaterThanOrEqual(2);
      const source = boxes.find(box => box.id === edge.source)!, target = boxes.find(box => box.id === edge.target)!;
      expect(route.points[0].x).toBe(source.x + source.width);
      expect(route.points.at(-1)!.x).toBe(target.x);
      for (let i = 1; i < route.points.length; i++) {
        const a = route.points[i - 1], b = route.points[i];
        for (const box of boxes) {
          const crosses = a.x === b.x
            ? a.x > box.x && a.x < box.x + box.width && Math.max(a.y, b.y) > box.y && Math.min(a.y, b.y) < box.y + box.height
            : a.y > box.y && a.y < box.y + box.height && Math.max(a.x, b.x) > box.x && Math.min(a.x, b.x) < box.x + box.width;
          expect(crosses, `${edge.id} crosses node ${box.id}`).toBe(false);
        }
      }
    }
    const trainInputs = layout.nodes["3"].ports.filter(port => port.side === "target");
    expect(new Set(trainInputs.map(port => port.y)).size).toBe(3);
    expect({ nodes, edges }).toEqual(before);
  });
});
