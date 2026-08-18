import { useState } from 'react';
import { api, button, card, fmtCost, fmtNum, fmtPct, fmtTime, input, label } from '../api';

const PIPELINES = {
  baseline: 'Vector search only. The comparison baseline.',
  hybrid: 'BM25 + vector search fused with Reciprocal Rank Fusion.',
  full: 'Hybrid retrieval + cross-encoder reranking.',
  reflective: 'If the answer is poorly grounded, retries with a broader retrieval strategy.',
  adaptive: 'Classifies the question (factoid / summary / comparison / temporal) and routes it.',
};
const CHOOSES_OWN_TOP_K = ['reflective', 'adaptive'];
const EXAMPLES = ['What were the main topics discussed?', 'What did the ECB decide about interest rates?', "What's the latest news?"];

const tone = (v) => (v >= 0.8 ? 'good' : v >= 0.5 ? 'mid' : 'bad');
const TONE_TEXT = { good: 'text-emerald-600', mid: 'text-amber-600', bad: 'text-red-600' };

function Faithfulness({ f }) {
  if (!f || !f.total_claims) return null;
  return (
    <div className={`${card} p-5 mt-4`}>
      <div className="flex items-center justify-between mb-3">
        <div className="font-medium text-slate-900">Faithfulness · {fmtPct(f.faithfulness_score)}</div>
        <div className="text-xs text-slate-500">{f.supported}/{f.total_claims} claims supported
          {f.contradicted > 0 && <span className="text-red-600"> · {f.contradicted} contradicted</span>}</div>
      </div>
      <div className="gauge-bar mb-4">
        <div className={`gauge-fill gauge-${tone(f.faithfulness_score)}`} style={{ width: `${f.faithfulness_score * 100}%` }} />
      </div>
      <div className="grid grid-cols-2 gap-3 mb-3 text-xs">
        {[['Numeric accuracy', f.numeric_accuracy], ['Citation accuracy', f.citation_accuracy]].map(([name, v]) => (
          <div key={name} className="bg-slate-50 rounded-md p-2">
            <div className="text-slate-500">{name}</div>
            <div className={`${TONE_TEXT[tone(v)]} font-medium`}>{fmtPct(v)}</div>
          </div>
        ))}
      </div>
      <details className="text-sm">
        <summary className="cursor-pointer text-xs text-slate-500">Per-claim verdicts</summary>
        {f.claim_results.map((c, i) => (
          <div key={i} className="flex items-start gap-2 py-1.5 border-b border-slate-100 last:border-0">
            <span className={`chip chip-${c.verdict} shrink-0`}>{c.verdict.replace('_', ' ')}</span>
            <span className="text-slate-700 flex-1">{c.claim}</span>
            <span className="text-xs text-slate-400 mono">{fmtNum(c.entailment_score)}</span>
          </div>
        ))}
      </details>
    </div>
  );
}

function Result({ data }) {
  const badges = [`pipeline: ${data.pipeline}`, data.query_type && `type: ${data.query_type}`,
    data.retry_count > 0 && `retries: ${data.retry_count}`, data.cached && 'served from cache'].filter(Boolean);
  return (
    <>
      <div className={`${card} p-5`}>
        <div className="flex flex-wrap gap-2 mb-3">{badges.map(b => <span key={b} className="pipeline-badge">{b}</span>)}</div>
        <div className="text-slate-800 leading-relaxed whitespace-pre-wrap">{data.answer}</div>
      </div>

      <Faithfulness f={data.faithfulness} />

      {data.sources.length > 0 && <div className={`${label} mt-4`}>Retrieved sources</div>}
      <div className="space-y-2">
        {data.sources.map(s => (
          <div key={s.chunk_id} className={`${card} p-3 text-sm`}>
            <div className="flex justify-between text-xs text-slate-500 mb-1.5 mono">
              <span>{s.chunk_id}{s.metadata.event_id != null && ` · story ${s.metadata.event_id}`}</span>
              <span>{fmtTime(s.metadata.start_time_utc)} · score {fmtNum(s.score, 3)}
                {s.recency_boost ? ` (+${fmtNum(s.recency_boost, 3)} recency)` : ''}</span>
            </div>
            <div className="text-slate-700">{s.text}</div>
          </div>
        ))}
      </div>

      <details className={`${card} p-4 mt-4`}>
        <summary className="text-xs font-medium text-slate-500 uppercase cursor-pointer">
          Trace · {fmtNum(data.trace.total_duration_ms, 0)} ms · {fmtCost(data.trace.total_cost_usd)}
        </summary>
        <table className="w-full text-sm mt-3">
          <tbody>
            {data.trace.spans.map((sp, i) => (
              <tr key={i}>
                <td className="py-1 mono text-slate-700">{sp.name}</td>
                <td className="py-1 text-right text-slate-600">{fmtNum(sp.duration_ms, 0)} ms</td>
                <td className="py-1 pl-4 text-xs text-slate-400">{sp.model || (sp.similarity && `similarity ${sp.similarity}`) || ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}

export default function QueryTab() {
  const [pipeline, setPipeline] = useState('full');
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(5);
  const [checkFaith, setCheckFaith] = useState(true);
  const [state, setState] = useState({});

  const run = async () => {
    if (!question.trim()) return;
    setState({ loading: true });
    try {
      const body = { question: question.trim(), pipeline, check_faithfulness: checkFaith };
      if (!CHOOSES_OWN_TOP_K.includes(pipeline)) body.top_k = topK;
      setState({ data: await api('/query', { body }) });
    } catch (e) {
      setState({ error: e.message });
    }
  };

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-5 space-y-4">
        <div className={`${card} p-5`}>
          <label className={label}>Pipeline</label>
          <select value={pipeline} onChange={e => setPipeline(e.target.value)} className={`${input} w-full`}>
            {Object.keys(PIPELINES).map(p => <option key={p}>{p}</option>)}
          </select>
          <p className="text-xs text-slate-500 mt-1.5">{PIPELINES[pipeline]}</p>

          <label className={`${label} mt-4`}>Question</label>
          <textarea rows={3} value={question} onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => (e.metaKey || e.ctrlKey) && e.key === 'Enter' && run()}
            className={`${input} w-full`} placeholder="What did the ECB decide about interest rates?" />

          <div className="flex items-center gap-4 mt-3 text-xs text-slate-600">
            <label className="flex items-center gap-1.5">
              <input type="checkbox" checked={checkFaith} onChange={e => setCheckFaith(e.target.checked)} />
              Faithfulness check
            </label>
            {!CHOOSES_OWN_TOP_K.includes(pipeline) && (
              <label className="flex items-center gap-1.5">top_k
                <input type="number" min={1} max={20} value={topK} className={`${input} w-16`}
                  onChange={e => setTopK(parseInt(e.target.value, 10) || 5)} />
              </label>
            )}
          </div>
          <button onClick={run} disabled={state.loading} className={`${button} w-full mt-4 py-2`}>
            {state.loading ? 'Running…' : 'Run query'}
          </button>
        </div>

        <div className={`${card} p-4 text-sm`}>
          <div className={label}>Try</div>
          {EXAMPLES.map(q => (
            <button key={q} onClick={() => setQuestion(q)} className="block text-left text-slate-700 hover:text-accent-700 py-0.5">{q}</button>
          ))}
        </div>
      </div>

      <div className="col-span-12 lg:col-span-7">
        {state.loading ? <div className={`${card} p-8 shimmer`} />
          : state.error ? <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">{state.error}</div>
          : state.data ? <Result data={state.data} />
          : <div className={`${card} p-12 text-center text-slate-400 text-sm`}>
              Ask a question to see the answer, its sources, the faithfulness breakdown and the pipeline trace.
            </div>}
      </div>
    </div>
  );
}
