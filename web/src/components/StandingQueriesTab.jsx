import { useEffect, useState } from 'react';
import { api, button, card, fmtNum, input } from '../api';

export default function StandingQueriesTab() {
  const [queries, setQueries] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [text, setText] = useState('');
  const [threshold, setThreshold] = useState(0.55);

  const load = () => {
    api('/standing-queries').then(d => setQueries(d.queries)).catch(() => {});
    api('/standing-queries/alerts').then(d => setAlerts(d.alerts)).catch(() => {});
  };
  useEffect(load, []);

  const create = async () => {
    if (!text.trim()) return;
    await api('/standing-queries', { body: { query: text.trim(), threshold } });
    setText('');
    load();
  };
  const remove = async (id) => {
    await api(`/standing-queries/${id}`, { method: 'DELETE' });
    load();
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-5 space-y-4">
        <div className={`${card} p-5`}>
          <h3 className="font-medium text-slate-900 mb-1">Create standing query</h3>
          <p className="text-xs text-slate-500 mb-3">
            Every newly ingested chunk is compared with this query; an alert fires when the cosine similarity reaches the threshold.
          </p>
          <input value={text} onChange={e => setText(e.target.value)} placeholder="e.g. interest rate decisions" className={`${input} w-full`} />
          <div className="flex items-center gap-3 mt-2">
            <label className="text-xs text-slate-600 flex items-center gap-2">threshold
              <input type="number" value={threshold} step={0.05} min={0.1} max={0.95} className={`${input} w-20`}
                onChange={e => setThreshold(parseFloat(e.target.value) || 0.55)} />
            </label>
            <button onClick={create} className={`${button} ml-auto`}>Create</button>
          </div>
        </div>

        <div className={`${card} p-5`}>
          <h3 className="font-medium text-slate-900 mb-3">Active queries</h3>
          {queries.length === 0 && <div className="text-xs text-slate-400">None yet.</div>}
          {queries.map(q => (
            <div key={q.id} className="flex items-center justify-between border border-slate-200 rounded-md px-3 py-2 mb-2">
              <div>
                <div className="text-sm text-slate-800">{q.query}</div>
                <div className="text-xs text-slate-500 mono">{q.id} · threshold {q.threshold} · {q.total_alerts} alerts</div>
              </div>
              <button onClick={() => remove(q.id)} className="text-xs text-slate-400 hover:text-red-600">Remove</button>
            </div>
          ))}
        </div>
      </div>

      <div className="col-span-12 lg:col-span-7">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-medium text-slate-900">Triggered alerts</h3>
          <button onClick={load} className="text-xs text-slate-500 hover:text-accent-700">Refresh</button>
        </div>
        {alerts.length === 0 && <div className={`${card} p-8 text-center text-sm text-slate-400`}>No alerts yet.</div>}
        <div className="space-y-2">
          {alerts.map((a, i) => (
            <div key={i} className={`${card} p-3 text-sm`}>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span className="mono">{a.chunk_id} · matched “{a.query_text}”</span>
                <span>similarity {fmtNum(a.similarity, 3)}</span>
              </div>
              <div className="text-slate-700">{a.preview}…</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
