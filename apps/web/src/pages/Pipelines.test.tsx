import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { expect, it } from "vitest";
import { Pipelines } from "./Pipelines";

it("redirects a historical Pipeline instance execution to the unified training detail", async () => {
  function Destination() { const location = useLocation(); return <p>{location.pathname}{location.search}</p>; }
  render(<MemoryRouter initialEntries={["/projects/p1/pipelines?tab=instances&instance=i1&run=r1"]}><Routes><Route path="/projects/:projectId"><Route path="pipelines" element={<Pipelines />} /><Route path="training" element={<Destination />} /></Route></Routes></MemoryRouter>);
  expect(await screen.findByText("/projects/p1/training?instance=i1&attempt=r1")).toBeInTheDocument();
});
