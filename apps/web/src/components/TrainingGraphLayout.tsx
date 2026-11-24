import { BaseEdge, Handle, Position, useNodeId, useUpdateNodeInternals, type Edge, type EdgeProps, type Node } from "@xyflow/react";
import { useEffect } from "react";
import type { ElkNode } from "elkjs/lib/elk-api";

export type DisplayPort = { id: string; side: "source" | "target"; x: number; y: number };
type Point = { x: number; y: number };
export type GraphLayout = {
  nodes: Record<string, { position: Point; height: number; ports: DisplayPort[] }>;
  edges: Record<string, { points: Point[]; sourceHandle: string; targetHandle: string }>;
};

export function DisplayHandles({ ports }: { ports: DisplayPort[] }) {
  const nodeId = useNodeId(), updateNodeInternals = useUpdateNodeInternals();
  const signature = JSON.stringify(ports);
  useEffect(() => { if (nodeId) updateNodeInternals(nodeId); }, [nodeId, signature, updateNodeInternals]);
  return <>{ports.map(port => <Handle key={port.id} id={port.id} type={port.side}
    position={port.side === "source" ? Position.Right : Position.Left} isConnectable={false}
    className="training-display-handle" style={{ left: port.x, right: "auto", top: port.y, transform: "translate(-50%, -50%)" }} />)}</>;
}

export function RoutedTrainingEdge({ id, sourceX, sourceY, targetX, targetY, data, markerEnd, style }: EdgeProps) {
  const routed = data?.points as Point[] | undefined;
  if (!routed?.length) return null;
  const points = [{ x: sourceX, y: sourceY }, ...routed.slice(1, -1), { x: targetX, y: targetY }];
  const path = points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />;
}

export async function layoutTrainingGraph(nodes: Node[], edges: Edge[]): Promise<GraphLayout> {
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  const nodeIds = new Set(nodes.map(node => node.id));
  const visibleEdges = edges.filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target));
  const result: GraphLayout = { nodes: {}, edges: {} };
  const input: ElkNode = {
    id: "training-layout",
    layoutOptions: {
      "elk.algorithm": "layered", "elk.direction": "RIGHT", "elk.edgeRouting": "ORTHOGONAL",
      "elk.spacing.nodeNode": "80", "elk.spacing.componentComponent": "120",
      "elk.layered.spacing.nodeNodeBetweenLayers": "180",
      "elk.layered.spacing.edgeNodeBetweenLayers": "32", "elk.spacing.edgeNode": "32",
      "elk.layered.spacing.edgeEdgeBetweenLayers": "22", "elk.spacing.edgeEdge": "22",
      "elk.layered.mergeEdges": "false", "elk.padding": "[top=40,left=40,bottom=40,right=40]",
    },
    children: nodes.map(node => {
      const incoming = visibleEdges.filter(edge => edge.target === node.id);
      const outgoing = visibleEdges.filter(edge => edge.source === node.id);
      const width = node.measured?.width || (node.type === "lineageContext" ? 245 : 260);
      const height = Math.max(node.measured?.height || 220, (Math.max(incoming.length, outgoing.length) + 1) * 22 + 24);
      const ports: DisplayPort[] = [];
      for (const [side, connected] of [["target", incoming], ["source", outgoing]] as const) {
        connected.forEach((edge, index) => ports.push({ id: `display:${edge.id}:${side}`, side, x: side === "source" ? width : 0, y: height * (index + 1) / (connected.length + 1) }));
      }
      result.nodes[node.id] = { position: node.position, height, ports };
      return { id: node.id, width, height, layoutOptions: { "elk.portConstraints": "FIXED_POS" },
        ports: ports.map(port => ({ id: port.id, x: port.x, y: port.y, width: 0, height: 0, layoutOptions: { "elk.port.side": port.side === "source" ? "EAST" : "WEST" } })) };
    }),
    edges: visibleEdges.map(edge => ({ id: edge.id, sources: [`display:${edge.id}:source`], targets: [`display:${edge.id}:target`] })),
  };
  const placed = await new ELK().layout(input);
  for (const node of placed.children || []) result.nodes[node.id].position = { x: node.x || 0, y: node.y || 0 };
  for (const edge of placed.edges || []) {
    const section = edge.sections?.[0];
    if (section) result.edges[edge.id] = { points: [section.startPoint, ...(section.bendPoints || []), section.endPoint], sourceHandle: `display:${edge.id}:source`, targetHandle: `display:${edge.id}:target` };
  }
  return result;
}
