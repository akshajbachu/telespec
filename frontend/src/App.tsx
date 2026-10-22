import { FormEvent, useMemo, useState } from "react";
import type { Finding, Result } from "./types";

type Status = "idle" | "working" | "error";

function linePath(x: number[], y: number[], width = 900, height = 230) {
  if (!x.length) return "";
  const xmin = Math.min(...x), xmax = Math.max(...x);
  const ymin = Math.min(...y), ymax = Math.max(...y);
  return x.map((value, i) => {
    const px = 20 + ((value - xmin) / Math.max(xmax - xmin, 1e-9)) * (width - 40);
    const py = 15 + (1 - (y[i] - ymin) / Math.max(ymax - ymin, 1e-9)) * (height - 30);
    return `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(" ");
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [column, setColumn] = useState("value");
  const [timestamp, setTimestamp] = useState("timestamp");
  const [trainFraction, setTrainFraction] = useState(0.3);
  const [fap, setFap] = useState(0.01);
  const [result, setResult] = useState<Result | null>(null);
  const [selected, setSelected] = useState<Finding | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return setError("Choose a CSV or JSONL telemetry file.");
    const body = new FormData();
    body.append("file", file); body.append("timestamp", timestamp); body.append("column", column);
    body.append("train_fraction", `${trainFraction}`); body.append("false_alarm_probability", `${fap}`);
    body.append("bootstraps", "100");
    await run("/api/analyze", { method: "POST", body });
  }

  async function run(url: string, init: RequestInit) {
    setStatus("working"); setError("");
    try {
      const response = await fetch(url, init);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Analysis failed");
      setResult(payload); setSelected(payload.findings[0] ?? null); setStatus("idle");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed"); setStatus("error");
    }
  }

  const signalPath = useMemo(
    () => result ? linePath(result.timestamps, result.values) : "", [result],
  );
  const spectrum = useMemo(
    () => result ? linePath(result.frequencies.map((f) => Math.log10(f)), result.power) : "",
    [result],
  );

  return <div className="shell">
    <header>
      <a className="brand" href="/"><span className="logo"><i/><i/><i/><i/></span>
        <span><b>tele</b>spec<small>network telemetry spectrum</small></span></a>
      <div className="local"><i/> LOCAL ANALYSIS <span>Your metrics stay on this device</span></div>
    </header>
    <main>
      <section className="hero">
        <div><p className="eyebrow">Spectral observability</p>
          <h1>See the cycles.<br/><em>Explain the change.</em></h1>
          <p>Detect hidden periodicity and spectral shifts in latency, traffic, jitter,
            loss, and throughput - without pretending irregular timestamps are regular.</p></div>
        <div className="hero-visual"><div className="rings"/><div className="beam"/>
          <span>GLS</span><small>bootstrap-calibrated</small></div>
      </section>

      <section className="workspace">
        <form className="panel controls" onSubmit={submit}>
          <div className="panel-title"><span>01</span><div><p>INPUT STREAM</p><h2>Load telemetry</h2></div></div>
          <label className={`dropzone ${file ? "chosen" : ""}`}>
            <input aria-label="Telemetry file" type="file" accept=".csv,.jsonl,.ndjson"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}/>
            <span className="upload">+</span><b>{file?.name ?? "Drop CSV or JSONL here"}</b>
            <small>{file ? `${(file.size / 1024).toFixed(1)} KB ready` : "or click to browse - 20 MB max"}</small>
          </label>
          <div className="field-row"><label>Timestamp column<input value={timestamp}
            onChange={(e) => setTimestamp(e.target.value)}/></label>
            <label>Metric column<input value={column} onChange={(e) => setColumn(e.target.value)}/></label></div>
          <label className="slider"><span>Training prefix <output>{Math.round(trainFraction * 100)}%</output></span>
            <input type="range" min=".1" max=".7" step=".05" value={trainFraction}
              onChange={(e) => setTrainFraction(Number(e.target.value))}/></label>
          <label className="slider"><span>False-alarm probability <output>{(fap * 100).toFixed(1)}%</output></span>
            <input type="range" min=".001" max=".05" step=".001" value={fap}
              onChange={(e) => setFap(Number(e.target.value))}/></label>
          {error && <div className="error" role="alert">{error}</div>}
          <button className="primary" disabled={status === "working"}>{status === "working" ? "Analyzing spectrum..." : "Analyze telemetry ->"}</button>
          <button className="demo" type="button" onClick={() => run("/api/demo", {method:"POST"})}>Load scientific demo</button>
        </form>

        <div className="results" aria-live="polite">
          {!result ? <section className="panel empty"><div className="empty-spectrum">
            {Array.from({length:36},(_,i)=><i key={i} style={{height:`${20 + Math.sin(i*.7)*14 + (i%7)*5}px`}}/>)}
            </div><p className="eyebrow">Awaiting signal</p><h2>Your spectral map starts here</h2>
            <p>Load a metric stream or run the demo to reveal statistically significant
              periods, change points, and anomaly evidence.</p>
            <div className="promises"><span>Irregular-time native</span><span>Global FAP</span><span>Explainable</span></div>
          </section> : <>
            <section className="result-head"><div><p className="eyebrow">Analysis complete</p>
              <h2>{result.filename}</h2><p>{result.sampling.count} samples - {(result.sampling.coverage*100).toFixed(1)}% coverage</p></div>
              <div className="exports"><a href={`/api/export/${result.id}?format=markdown`}>Report</a>
                <a href={`/api/export/${result.id}?format=json`}>JSON</a></div></section>
            <section className="kpis">
              <article><span>Findings</span><b>{result.findings.length}</b></article>
              <article><span>Significant cycles</span><b>{result.peaks.length}</b></article>
              <article><span>Sampling jitter</span><b>{result.sampling.jitter_mad.toFixed(2)}<small>s</small></b></article>
              <article><span>Longest period</span><b>{result.peaks[0] ? humanPeriod(result.peaks[0].period) : "-"}</b></article>
            </section>
            <Chart title="Telemetry timeline" subtitle="Finding intervals are marked as evidence overlays" path={signalPath}
              findings={result.findings} times={result.timestamps}/>
            <Chart title="Generalized Lomb-Scargle periodogram" subtitle={`Global bootstrap threshold ${result.threshold_power.toFixed(3)}`}
              path={spectrum} threshold={result.threshold_power}/>
            <Spectrogram
              times={result.metadata.spectrogram.times}
              frequencies={result.metadata.spectrogram.frequencies}
              power={result.metadata.spectrogram.power}
            />
            <section className="panel finding-panel"><div className="panel-title"><span>04</span><div><p>ATTRIBUTION</p><h2>Interpretable findings</h2></div></div>
              <div className="finding-grid"><div className="finding-list">
                {result.findings.map((finding,i)=><button className={selected===finding?"active":""}
                  key={`${finding.kind}-${i}`} onClick={()=>setSelected(finding)}>
                  <i className={finding.kind}/><span><b>{finding.kind.replaceAll("_"," ")}</b>
                  <small>{(finding.confidence*100).toFixed(0)}% confidence</small></span><em>+</em></button>)}
                {!result.findings.length && <p>No findings cleared the configured evidence thresholds.</p>}
              </div>{selected && <article className="evidence"><p className="eyebrow">Evidence drawer</p>
                <h3>{selected.kind.replaceAll("_"," ")}</h3><p>{selected.summary}</p>
                <dl><div><dt>Magnitude</dt><dd>{selected.magnitude.toFixed(4)}</dd></div>
                  <div><dt>Confidence</dt><dd>{(selected.confidence*100).toFixed(1)}%</dd></div>
                  <div><dt>Interval</dt><dd>{selected.start.toFixed(0)} - {selected.end.toFixed(0)}</dd></div>
                  {selected.period && <div><dt>Period</dt><dd>{humanPeriod(selected.period)}</dd></div>}</dl>
              </article>}</div>
            </section>
          </>}
        </div>
      </section>
    </main><footer><span>telespec v0.1.0 - research-grade local analysis</span>
      <span>GLS / CUSUM / robust covariance</span></footer>
  </div>;
}

function Chart({title,subtitle,path,threshold,findings,times}:{title:string;subtitle:string;path:string;
  threshold?:number;findings?:Finding[];times?:number[]}) {
  const start=times?.[0]??0, span=(times?.at(-1)??1)-start;
  return <section className="panel chart"><div className="chart-head"><div><p>{subtitle}</p><h2>{title}</h2></div><span>LOG POWER</span></div>
    <svg viewBox="0 0 900 230" role="img" aria-label={title}>
      <defs><linearGradient id="line" x1="0" x2="1"><stop stopColor="#27e0c1"/><stop offset="1" stopColor="#7c5cff"/></linearGradient></defs>
      {[1,2,3,4].map(i=><line key={i} x1="20" y1={i*45} x2="880" y2={i*45}/>)}
      {findings?.map((f,i)=><rect key={i} className="overlay" x={20+((f.start-start)/span)*860}
        y="15" width={Math.max(3,((f.end-f.start)/span)*860)} height="200"/>)}
      {threshold!==undefined && <line className="threshold" x1="20" y1={215-threshold*190} x2="880" y2={215-threshold*190}/>}
      <path className="data-line" d={path}/></svg></section>;
}

function Spectrogram({times,frequencies,power}:{times:number[];frequencies:number[];power:number[][]}) {
  const columns=Math.max(times.length,1), rows=Math.max(frequencies.length,1);
  const transformed=power.flat().map((value)=>Math.log10(value+1e-12));
  const low=Math.min(...transformed), high=Math.max(...transformed);
  return <section className="panel chart"><div className="chart-head"><div>
    <p>STFT on explicitly resampled, gap-checked data</p><h2>Spectral evolution</h2>
    </div><span>TIME / FREQUENCY</span></div>
    <svg className="heatmap" viewBox="0 0 900 230" role="img" aria-label="Spectral evolution spectrogram">
      {power.flatMap((row,rowIndex)=>row.map((value,columnIndex)=>{
        const normalized=(Math.log10(value+1e-12)-low)/Math.max(high-low,1e-9);
        return <rect key={`${rowIndex}-${columnIndex}`} x={columnIndex*900/columns}
          y={(rows-rowIndex-1)*230/rows} width={900/columns+0.4} height={230/rows+0.4}
          fill={`hsl(${255-normalized*90} 75% ${10+normalized*55}%)`}/>;
      }))}
    </svg></section>;
}

function humanPeriod(seconds:number) {
  return seconds>=86400?`${(seconds/86400).toFixed(1)}d`:seconds>=3600?`${(seconds/3600).toFixed(1)}h`:
    seconds>=60?`${(seconds/60).toFixed(1)}m`:`${seconds.toFixed(1)}s`;
}
export default App;
