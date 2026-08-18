import { useEffect, useState } from 'react';
import { api, button, card, fmtNum, input } from '../api';

const LOWER_IS_BETTER = /latency|cost/;

function BenchmarkTable({ data }) {
  const { pipelines, comparison_table: rows, pareto_optimal: pareto } = data;
  const groups = rows.reduce((g, r) => ({ ...g, [r.category]: [...(g[r.category] || []), r] }), {});

  return (
    <>
      <div className="mb-3 text-xs text-slate-500">
        <span className="text-accent-700 font-semibold">Bold</span> = best per metric ·
        ★ = Pareto-optimal (no other pipeline is at least as faithful, as fast and as cheap)
      </div>
      {Object.entries(groups).map(([category, metrics]) => (
        <div key={category} className={`${card} overflow-hidden mb-4`}>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-slate-500 bg-slate-50 border-b border-slate-200">
                <th className="px-3 py-2 text-left font-medium uppercase">{category}</th>
                {pipelines.map(p => <th key={p} className="px-3 py-2 text-right font-normal mono">{p}{pareto.includes(p) && ' ★'}</th>)}
              </tr>
            </thead>
            <tbody>
              {metrics.map(row => {
                const values = pipelines.map(p => row[p]);
                const present = values.filter(v => v != null);
                const best = LOWER_IS_BETTER.test(row.metric) ? Math.min(...present) : Math.max(...present);
                const tie = present.every(v => v === best);
                return (
                  <tr key={row.metric} className="border-b border-slate-100 last:border-0">
                    <td className="px-3 py-1.5 text-slate-700">{row.metric}</td>
                    {values.map((v, i) => (
                      <td key={i} className={`px-3 py-1.5 text-right mono ${v === best && !tie ? 'text-accent-700 font-semibold' : 'text-slate-700'}`}>
                        {fmtNum(v, row.metric.includes('cost') ? 6 : 3)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}

export default function EvaluationTab() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset] = useState('sample');
  const [judge, setJudge] = useState(false);
  const [state, setState] = useState({});

  useEffect(() => { api('/datasets').then(d => setDatasets(d.datasets)).catch(() => {}); }, []);

  const run = async () => {
    setState({ loading: true });
    try {
      setState({ data: await api('/eval/benchmark', { body: { dataset_id: dataset, judge } }) });
    } catch (e) {
      setState({ error: e.message });
    }
  };

  return (
    <>
      <div className={`${card} p-5 mb-6`}>
        <h3 className="font-medium text-slate-900 mb-1">Benchmark pipelines</h3>
        <p className="text-xs text-slate-500 mb-3">
          Runs every question in the dataset through baseline, hybrid, full and adaptive on a separate evaluation index
          (your live index is untouched), then compares retrieval quality, faithfulness, latency and cost.
          Runs are saved for <span className="mono">python -m app.cli compare / pareto</span>.
        </p>
        <div className="flex items-center gap-3">
          <select value={dataset} onChange={e => setDataset(e.target.value)} className={input}>
            {datasets.map(d => <option key={d.dataset_id} value={d.dataset_id}>{d.dataset_id} ({d.num_qa_pairs} questions)</option>)}
          </select>
          <label className="text-xs text-slate-600 flex items-center gap-1.5">
            <input type="checkbox" checked={judge} onChange={e => setJudge(e.target.checked)} />
            LLM-as-judge (extra API calls)
          </label>
          <button onClick={run} disabled={state.loading} className={`${button} ml-auto`}>
            {state.loading ? 'Running… (about a minute per pipeline)' : 'Run benchmark'}
          </button>
        </div>
      </div>
      {state.error && <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">{state.error}</div>}
      {state.loading && <div className={`${card} p-8 shimmer`} />}
      {state.data && <BenchmarkTable data={state.data} />}
    </>
  );
}
