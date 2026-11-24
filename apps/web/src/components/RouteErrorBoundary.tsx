import { Component, type ErrorInfo, type ReactNode } from "react";
import { useLocation } from "react-router-dom";

type BoundaryProps = {
  children: ReactNode;
  resetKey: string;
  systemPage?: boolean;
};

type BoundaryState = {
  error: Error | null;
  resetKey: string;
};

class RouteContentErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null, resetKey: this.props.resetKey };

  static getDerivedStateFromProps(props: BoundaryProps, state: BoundaryState): Partial<BoundaryState> | null {
    return props.resetKey !== state.resetKey ? { error: null, resetKey: props.resetKey } : null;
  }

  static getDerivedStateFromError(error: Error): Partial<BoundaryState> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Route content render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const content = <><span>页面渲染失败</span><h1>这个页面遇到了异常</h1><p>{this.state.error.message || "页面内容无法继续渲染。"}</p><div><button type="button" className="button secondary" onClick={() => this.setState({ error: null })}>重试页面</button><button type="button" className="button primary" onClick={() => window.location.reload()}>刷新应用</button></div></>;
    return this.props.systemPage
      ? <main className="system-page route-error" role="alert">{content}</main>
      : <section className="route-error" role="alert">{content}</section>;
  }
}

export function RouteErrorBoundary({ children, systemPage = false }: { children: ReactNode; systemPage?: boolean }) {
  const location = useLocation();
  return <RouteContentErrorBoundary resetKey={`${location.pathname}${location.search}`} systemPage={systemPage}>{children}</RouteContentErrorBoundary>;
}
