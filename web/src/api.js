export async function api(path, { body, ...opts } = {}) {
  const res = await fetch('/api/v1' + path, {
    headers: { 'Content-Type': 'application/json' },
    method: body ? 'POST' : 'GET',
    body: body && JSON.stringify(body),
    ...opts,
  });
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

export const fmtPct = (n) => (n == null ? '—' : Math.round(n * 100) + '%');
export const fmtNum = (n, d = 2) => (n == null ? '—' : Number(n).toFixed(d));
export const fmtCost = (n) => (n == null ? '—' : '$' + Number(n).toFixed(5));
export const fmtTime = (iso) => (iso ? iso.slice(11, 16) + ' UTC' : '');

export const card = 'bg-white border border-slate-200 rounded-lg';
export const input = 'bg-slate-50 border border-slate-200 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-accent-500';
export const button = 'bg-accent-600 hover:bg-accent-700 text-white text-sm px-4 py-1.5 rounded-md disabled:opacity-50';
export const label = 'block text-xs font-medium text-slate-500 uppercase tracking-wide mb-2';
