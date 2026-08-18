import { useEffect, useRef, useState } from 'react';
import { api, button, card, fmtTime, input } from '../api';

function LiveCapture() {
  const audio = useRef(null);
  const [status, setStatus] = useState({ running: false, recent: [] });
  const [listening, setListening] = useState(false);

  // Poll capture status while this tab is open.
  useEffect(() => {
    const poll = () => api('/ingest/status').then(setStatus).catch(() => {});
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  const toggleListen = () => {
    if (listening) { audio.current.pause(); audio.current.src = ''; }
    else { audio.current.src = '/stream-proxy'; audio.current.play().catch(() => setListening(false)); }
    setListening(!listening);
  };
  const capture = async (action) => setStatus(await api(`/ingest/${action}`, { method: 'POST' }));

  return (
    <div className={`${card} p-5`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={`w-3 h-3 rounded-full ${status.running ? 'bg-red-500 pulse-live' : 'bg-slate-300'}`} />
        <h3 className="font-medium text-slate-900">Live stream · BBC World Service</h3>
      </div>
      <p className="text-xs text-slate-500 mb-4">
        Capture records the stream in 60-second chunks, transcribes them with Whisper, splits them by topic and indexes them.
        Each new chunk is matched against stories, standing queries and the answer cache.
      </p>
      <audio ref={audio} preload="none" />
      <div className="flex items-center gap-3">
        <button onClick={toggleListen} className="bg-slate-100 hover:bg-slate-200 text-slate-700 text-sm px-4 py-1.5 rounded-md">
          {listening ? 'Stop listening' : 'Listen in browser'}
        </button>
        {status.running || status.stopping
          ? <button onClick={() => capture('stop')} disabled={status.stopping}
              className="bg-red-100 hover:bg-red-200 text-red-700 text-sm px-4 py-1.5 rounded-md disabled:opacity-50">
              {status.stopping ? 'Finishing current chunk…' : 'Stop capture'}
            </button>
          : <button onClick={() => capture('start')} className={button}>Start capture</button>}
      </div>

      <div className="mt-4 space-y-2 max-h-96 overflow-y-auto">
        {status.recent.length === 0 && <div className="text-xs text-slate-400">No chunks captured yet.</div>}
        {status.recent.map((r, i) => r.error ? (
          <div key={i} className="text-xs text-red-700 bg-red-50 rounded px-3 py-2">{fmtTime(r.time)} · {r.error}</div>
        ) : (
          <div key={i} className="bg-slate-50 border border-slate-100 rounded-lg p-3 text-sm">
            <div className="flex justify-between text-xs text-slate-500 mb-1 mono">
              <span>{r.chunk_id} · {r.indexed_ids.length} chunk(s) · {r.word_count} words</span>
              <span>{fmtTime(r.time)}</span>
            </div>
            <div className="text-slate-700">{r.preview}…</div>
            <div className="flex flex-wrap gap-2 mt-2 text-xs">
              {r.new_events.map(e => <span key={e} className="pipeline-badge">new story {e}</span>)}
              {r.cache_evictions > 0 && <span className="pipeline-badge">{r.cache_evictions} cached answer(s) refreshed</span>}
              {r.alerts.map(a => <span key={a.query_id} className="text-amber-700 bg-amber-50 rounded px-2">alert: “{a.query_text}” ({a.similarity})</span>)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TextIngest() {
  const [text, setText] = useState('');
  const [source, setSource] = useState('');
  const [result, setResult] = useState('');

  const ingest = async () => {
    if (!text.trim()) return;
    setResult('Ingesting…');
    try {
      const r = await api('/ingest/text', { body: { text: text.trim(), source: source.trim() || 'manual' } });
      setResult(`Indexed ${r.indexed_ids.length} chunk(s), ${r.word_count} words` +
        (r.alerts.length ? ` · ${r.alerts.length} alert(s) fired` : '') +
        (r.cache_evictions ? ` · ${r.cache_evictions} cached answer(s) refreshed` : ''));
      setText('');
    } catch (e) {
      setResult(`Error: ${e.message}`);
    }
  };

  return (
    <div className={`${card} p-5`}>
      <h3 className="font-medium text-slate-900 mb-1">Ingest text</h3>
      <p className="text-xs text-slate-500 mb-3">Paste any transcript. It goes through the same pipeline as live audio, minus the transcription.</p>
      <textarea rows={8} value={text} onChange={e => setText(e.target.value)} className={`${input} w-full`}
        placeholder="Paste a news transcript here…" />
      <div className="flex gap-3 mt-2">
        <input value={source} onChange={e => setSource(e.target.value)} placeholder="source name" className={`${input} flex-1`} />
        <button onClick={ingest} className={button}>Ingest</button>
      </div>
      {result && <div className="mt-2 text-xs text-slate-600">{result}</div>}
    </div>
  );
}

export default function IngestTab() {
  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-7"><LiveCapture /></div>
      <div className="col-span-12 lg:col-span-5"><TextIngest /></div>
    </div>
  );
}
