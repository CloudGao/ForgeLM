export function RouteLoading({ label = "正在加载页面内容…" }: { label?: string }) {
  return <div className="route-loading" role="status" aria-label={label} aria-live="polite"><span aria-hidden="true" /><strong>{label}</strong></div>;
}
