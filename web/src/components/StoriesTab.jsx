import { useEffect, useState } from 'react';
import { api, card, fmtTime } from '../api';

export default function StoriesTab() {
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState([]);

  const load = () => {
    api('/stats').then(setStats).catch(() => {});
    api('/events').then(d => setEvents(d.events)).catch(() => {});
  };
  useEffect(load, []);

  const tiles = stats ? [
    ['Chunks', stats.total_docs],
    ['Words', stats.total_words.toLocaleString()],
    ['Time span', `${Math.round(stats.span_minutes)} min`],
    ['Stories', events.length],
    ['Cache hits / misses', `${stats.cache.hits} / ${stats.cache.misses}`],
    ['Cache refreshes', stats.cache.invalidated],
  ] : [];

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-6">
        {tiles.map(([name, value]) => (
          <div key={name} className={`${card} p-4`}>
            <div className="text-xs text-slate-500 uppercase tracking-wide">{name}</div>
            <div className="text-2xl font-semibold text-slate-900 mt-1">{value}</div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-slate-900">Stories <span className="text-xs text-slate-500 font-normal">
          · consecutive chunks with similar meaning are grouped into one developing story</span></h3>
        <button onClick={load} className="text-xs text-slate-500 hover:text-accent-700">Refresh</button>
      </div>
      {events.length === 0
        ? <div className={`${card} p-8 text-center text-sm text-slate-400`}>Nothing indexed yet. Capture or ingest some content first.</div>
        : <div className="space-y-2">
            {events.map(e => (
              <div key={e.event_id} className={`${card} p-3 text-sm`}>
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span className="pipeline-badge">story {e.event_id}</span>
                  <span>{fmtTime(e.start_time)} · {e.num_chunks} chunk(s) · {e.duration_minutes} min</span>
                </div>
                <div className="text-slate-700">{e.preview}…</div>
              </div>
            ))}
          </div>}
    </>
  );
}
