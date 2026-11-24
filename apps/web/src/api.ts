const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export const apiHref = (path: string) => `${API_URL}/api${path}`;

export class ApiError extends Error {
  status: number;
  detail: any;

  constructor(status: number, detail: any) {
    super(detail?.message || detail?.detail?.message || `请求失败 (${status})`);
    this.status = status;
    this.detail = detail?.detail ?? detail;
  }
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const isFormData = typeof FormData !== "undefined" && options?.body instanceof FormData;
  const response = await fetch(`${API_URL}/api${path}`, {
    ...options,
    headers: {
      ...(!isFormData ? { "Content-Type": "application/json" } : {}),
      ...(options?.headers || {}),
    },
  });
  if (!response.ok) {
    let detail: any = {};
    try {
      detail = await response.json();
    } catch {
      detail = { message: response.statusText };
    }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const post = <T>(path: string, data?: unknown) =>
  api<T>(path, { method: "POST", body: JSON.stringify(data ?? {}) });

export const patch = <T>(path: string, data: unknown) =>
  api<T>(path, { method: "PATCH", body: JSON.stringify(data) });

export const postForm = <T>(path: string, data: FormData) =>
  api<T>(path, { method: "POST", body: data });

export function formatNumber(value?: number): string {
  if (value == null) return "—";
  return new Intl.NumberFormat("zh-CN", { notation: value > 9999 ? "compact" : "standard", maximumFractionDigits: 1 }).format(value);
}

export function formatDate(value?: string): string {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

export function formatBytes(value?: number | null): string {
  if (value == null) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = Math.max(Number(value) || 0, 0);
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1; }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
}
