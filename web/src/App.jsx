import { useEffect, useState } from 'react';
import { api } from './api';
import QueryTab from './components/QueryTab';
import IngestTab from './components/IngestTab';
import StoriesTab from './components/StoriesTab';
import StandingQueriesTab from './components/StandingQueriesTab';
import EvaluationTab from './components/EvaluationTab';

const TABS = [
  ['query', 'Query', QueryTab],
  ['ingest', 'Ingest', IngestTab],
  ['stories', 'Stories', StoriesTab],
  ['alerts', 'Standing Queries', StandingQueriesTab],
  ['eval', 'Evaluation', EvaluationTab],
];

function useHealth() {
  const [health, setHealth] = useState(null);
  useEffect(() => {
    const check = () => api('/health').then(setHealth).catch(() => setHealth(null));
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, []);
  return health;
}

export default function App() {
  const [tab, setTab] = useState('query');
  const health = useHealth();
  const Active = TABS.find(([id]) => id === tab)[2];

  return (
    <>
      <header className="border-b border-slate-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <div className="font-semibold text-slate-900">Streaming RAG</div>
            <div className="text-xs text-slate-500">Ask questions about live radio, with answers checked against the sources</div>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-600">
            <span className={`w-2 h-2 rounded-full ${health ? 'bg-emerald-500' : 'bg-red-500'}`} />
            {health ? `${health.indexed_chunks} chunks · ${health.llm_provider} · v${health.version}` : 'offline'}
          </div>
        </div>
        <nav className="max-w-7xl mx-auto px-6 flex gap-6 text-sm">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)}
              className={`py-3 border-b-2 ${tab === id
                ? 'border-accent-600 text-accent-700 font-medium'
                : 'border-transparent text-slate-500 hover:text-slate-700'}`}>
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="max-w-7xl mx-auto px-6 py-8 tab-panel" key={tab}>
        <Active />
      </main>
    </>
  );
}
