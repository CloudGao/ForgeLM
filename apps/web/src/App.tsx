import { lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { ProjectShell } from "./components/ProjectShell";
import { SystemShell } from "./components/SystemShell";

const ProjectCenter = lazy(() => import("./pages/ProjectCenter").then((module) => ({ default: module.ProjectCenter })));
const Home = lazy(() => import("./pages/Home").then((module) => ({ default: module.Home })));
const Overview = lazy(() => import("./pages/Overview").then((module) => ({ default: module.Overview })));
const Tokenizers = lazy(() => import("./pages/Tokenizers").then((module) => ({ default: module.Tokenizers })));
const Training = lazy(() => import("./pages/Training").then((module) => ({ default: module.Training })));
const Models = lazy(() => import("./pages/Models").then((module) => ({ default: module.Models })));
const Evaluations = lazy(() => import("./pages/Evaluations").then((module) => ({ default: module.Evaluations })));
const Releases = lazy(() => import("./pages/Releases").then((module) => ({ default: module.Releases })));
const Playground = lazy(() => import("./pages/Playground").then((module) => ({ default: module.Playground })));
const ExecutionEnvironments = lazy(() => import("./pages/ExecutionEnvironments").then((module) => ({ default: module.ExecutionEnvironments })));
const PluginCenter = lazy(() => import("./pages/PluginCenter").then((module) => ({ default: module.PluginCenter })));
const StorageCenter = lazy(() => import("./pages/StorageCenter").then((module) => ({ default: module.StorageCenter })));
const GlobalDataCenter = lazy(() => import("./pages/GlobalDataCenter").then((module) => ({ default: module.GlobalDataCenter })));
const ProjectDataBindings = lazy(() => import("./pages/ProjectDataBindings").then((module) => ({ default: module.ProjectDataBindings })));
const BenchmarkCenter = lazy(() => import("./pages/BenchmarkCenter").then((module) => ({ default: module.BenchmarkCenter })));
const ModelWeights = lazy(() => import("./pages/ModelWeights").then((module) => ({ default: module.ModelWeights })));
const Pipelines = lazy(() => import("./pages/Pipelines").then((module) => ({ default: module.Pipelines })));

export function App() {
  return (
    <Routes>
      <Route element={<SystemShell />}>
        <Route path="/" element={<Home />} />
        <Route path="/projects" element={<ProjectCenter />} />
        <Route path="/execution" element={<ExecutionEnvironments />} />
        <Route path="/execution/plugins" element={<PluginCenter />} />
        <Route path="/storage" element={<StorageCenter />} />
        <Route path="/data" element={<GlobalDataCenter />} />
        <Route path="/benchmarks" element={<BenchmarkCenter />} />
        <Route path="/weights" element={<ModelWeights />} />
      </Route>
      <Route path="/projects/:projectId" element={<ProjectShell />}>
        <Route index element={<Navigate to="overview" replace />} />
        <Route path="overview" element={<Overview />} />
        <Route path="pipelines" element={<Pipelines />} />
        <Route path="data" element={<ProjectDataBindings />} />
        <Route path="tokenizers" element={<Tokenizers />} />
        <Route path="training" element={<Training />} />
        <Route path="models" element={<Models />} />
        <Route path="evaluations" element={<Evaluations />} />
        <Route path="releases" element={<Releases />} />
        <Route path="playground" element={<Playground />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
