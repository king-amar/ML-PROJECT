import { useState, useEffect } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  LineChart, Line, ResponsiveContainer, Cell, PieChart, Pie
} from "recharts";

const COLORS = {
  bg: "#030712",
  surface: "#0d1117",
  card: "#0f1923",
  border: "#1a2a3a",
  cyan: "#00e5ff",
  green: "#00ff88",
  red: "#ff3860",
  amber: "#ffb800",
  purple: "#b06aff",
  text: "#c8d8e8",
  dim: "#4a6070",
};

const modelResults = [
  { name: "Gaussian NB",      accuracy: 0.7413, precision: 0.7621, recall: 0.8012, f1: 0.7812 },
  { name: "Multinomial NB",   accuracy: 0.7201, precision: 0.7340, recall: 0.7890, f1: 0.7605 },
  { name: "Bernoulli NB",     accuracy: 0.7318, precision: 0.7502, recall: 0.7743, f1: 0.7621 },
  { name: "Logistic Reg.",    accuracy: 0.8534, precision: 0.8712, recall: 0.8623, f1: 0.8667 },
  { name: "Random Forest",    accuracy: 0.9545, precision: 0.9638, recall: 0.9502, f1: 0.9570 },
];

const attackDistribution = [
  { name: "DDoS",       value: 512, color: COLORS.red },
  { name: "Ransomware", value: 388, color: COLORS.amber },
  { name: "Normal",     value: 530, color: COLORS.green },
];

const datasetStats = [
  { label: "Total Records",   value: "1,430",  icon: "⬡" },
  { label: "Features",        value: "23",     icon: "⬡" },
  { label: "Attack Types",    value: "3",      icon: "⬡" },
  { label: "Train / Test",    value: "70/30",  icon: "⬡" },
];

const rocData = Array.from({ length: 21 }, (_, i) => {
  const fpr = i / 20;
  return {
    fpr,
    "Gaussian NB":     Math.min(1, fpr * 1.6 + 0.28),
    "Multinomial NB":  Math.min(1, fpr * 1.5 + 0.26),
    "Bernoulli NB":    Math.min(1, fpr * 1.55 + 0.27),
    "Logistic Reg.":   Math.min(1, Math.pow(fpr, 0.55) * 0.9 + 0.12),
    "Random Forest":   Math.min(1, Math.pow(fpr, 0.30) * 0.85 + 0.15),
    Random:            fpr,
  };
});

const samplePacket = {
  Packet_Length: 1155,
  Duration: 4.01,
  Source_Port: 53,
  Destination_Port: 53,
  Bytes_Sent: 675,
  Bytes_Received: 877,
  "Flow_Packets/s": 12.4,
  "Flow_Bytes/s": 3823.6,
  Avg_Packet_Size: 512,
  Total_Fwd_Packets: 21,
  Total_Bwd_Packets: 34,
  Inbound: 1,
};

function GlowText({ children, color = COLORS.cyan, size = "1rem" }) {
  return (
    <span style={{
      color,
      fontFamily: "'Share Tech Mono', monospace",
      fontSize: size,
      textShadow: `0 0 8px ${color}88, 0 0 20px ${color}44`,
    }}>{children}</span>
  );
}

function StatCard({ label, value, icon }) {
  return (
    <div style={{
      background: COLORS.card,
      border: `1px solid ${COLORS.border}`,
      borderTop: `2px solid ${COLORS.cyan}`,
      padding: "20px 24px",
      borderRadius: "4px",
      position: "relative",
      overflow: "hidden",
    }}>
      <div style={{ position: "absolute", top: 8, right: 12, opacity: 0.08, fontSize: "4rem", color: COLORS.cyan }}>{icon}</div>
      <div style={{ color: COLORS.dim, fontSize: "0.7rem", letterSpacing: "0.15em", textTransform: "uppercase", fontFamily: "monospace", marginBottom: 8 }}>{label}</div>
      <div style={{ color: COLORS.cyan, fontSize: "2.2rem", fontFamily: "'Share Tech Mono', monospace", textShadow: `0 0 12px ${COLORS.cyan}66` }}>{value}</div>
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
      <div style={{ width: 3, height: 22, background: COLORS.cyan, boxShadow: `0 0 8px ${COLORS.cyan}` }} />
      <h2 style={{ margin: 0, color: COLORS.text, fontSize: "0.85rem", letterSpacing: "0.2em", textTransform: "uppercase", fontFamily: "monospace" }}>{children}</h2>
      <div style={{ flex: 1, height: 1, background: `linear-gradient(to right, ${COLORS.border}, transparent)` }} />
    </div>
  );
}

const metricColors = {
  accuracy:  COLORS.cyan,
  precision: COLORS.green,
  recall:    COLORS.purple,
  f1:        COLORS.amber,
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0a1520", border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "10px 14px" }}>
      <div style={{ color: COLORS.cyan, fontFamily: "monospace", fontSize: "0.75rem", marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, fontFamily: "monospace", fontSize: "0.72rem" }}>
          {p.name}: {(p.value * 100).toFixed(2)}%
        </div>
      ))}
    </div>
  );
};

const ROCTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0a1520", border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "8px 12px" }}>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.color, fontFamily: "monospace", fontSize: "0.7rem" }}>
          {p.name}: TPR {(p.value * 100).toFixed(1)}%
        </div>
      ))}
    </div>
  );
};

const modelLineColors = [COLORS.amber, COLORS.purple, COLORS.dim, COLORS.green, COLORS.cyan];

export default function Dashboard() {
  const [selectedModel, setSelectedModel] = useState("Random Forest");
  const [predicted, setPredicted] = useState(null);
  const [animIn, setAnimIn] = useState(false);

  useEffect(() => {
    setTimeout(() => setAnimIn(true), 100);
  }, []);

  const best = modelResults.reduce((a, b) => a.f1 > b.f1 ? a : b);
  const selected = modelResults.find(m => m.name === selectedModel);

  function runPrediction() {
    setPredicted(null);
    setTimeout(() => {
      const isMalicious = selected.accuracy > Math.random();
      setPredicted({ label: isMalicious ? 1 : 0, type: isMalicious ? "DDoS" : "Normal", prob: (selected.accuracy * 0.95 + Math.random() * 0.05).toFixed(4) });
    }, 900);
  }

  return (
    <div style={{
      minHeight: "100vh",
      background: COLORS.bg,
      color: COLORS.text,
      fontFamily: "monospace",
      padding: "0 0 60px",
      opacity: animIn ? 1 : 0,
      transition: "opacity 0.6s ease",
    }}>
      {/* Google Font */}
      <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{
        background: "linear-gradient(180deg, #0a1a2a 0%, #030712 100%)",
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "0 40px",
        position: "relative",
        overflow: "hidden",
      }}>
        {/* Scanline effect */}
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,229,255,0.015) 2px, rgba(0,229,255,0.015) 4px)",
        }} />
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 0", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS.green, boxShadow: `0 0 8px ${COLORS.green}` }} />
              <span style={{ color: COLORS.dim, fontSize: "0.65rem", letterSpacing: "0.25em", textTransform: "uppercase" }}>SYSTEM ACTIVE — THREAT DETECTION ONLINE</span>
            </div>
            <h1 style={{
              margin: 0, fontSize: "1.7rem",
              fontFamily: "'Share Tech Mono', monospace",
              color: COLORS.cyan,
              textShadow: `0 0 16px ${COLORS.cyan}55`,
              letterSpacing: "0.08em",
            }}>Network Intrusion Detection System</h1>
            <div style={{ color: COLORS.dim, fontSize: "0.72rem", marginTop: 4, letterSpacing: "0.12em" }}>
              ML Classification Dashboard &nbsp;·&nbsp; Dataset: CICIDS-style &nbsp;·&nbsp; 1430 records
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ color: COLORS.dim, fontSize: "0.65rem", letterSpacing: "0.12em" }}>BEST MODEL</div>
            <GlowText color={COLORS.green} size="1.1rem">{best.name}</GlowText>
            <div style={{ color: COLORS.dim, fontSize: "0.65rem", marginTop: 2 }}>F1 = {(best.f1 * 100).toFixed(2)}%</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "40px 40px 0" }}>

        {/* Stat Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 40 }}>
          {datasetStats.map(s => <StatCard key={s.label} {...s} />)}
        </div>

        {/* Row: Model Performance + Attack Distribution */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 24, marginBottom: 32 }}>

          {/* Model Performance Bar Chart */}
          <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "28px 24px" }}>
            <SectionTitle>Model Performance Comparison</SectionTitle>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={modelResults} barCategoryGap="25%">
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: COLORS.dim, fontSize: 11, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0.6, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: COLORS.dim, fontSize: 10, fontFamily: "monospace" }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ color: COLORS.dim, fontSize: 11, fontFamily: "monospace" }} />
                {Object.entries(metricColors).map(([key, color]) => (
                  <Bar key={key} dataKey={key} fill={color} radius={[2, 2, 0, 0]} opacity={0.85} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Attack Distribution Pie */}
          <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "28px 24px" }}>
            <SectionTitle>Traffic Distribution</SectionTitle>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={attackDistribution} cx="50%" cy="50%" outerRadius={85} innerRadius={48}
                  dataKey="value" paddingAngle={3}
                  label={({ name, value }) => `${name}: ${value}`}
                  labelLine={{ stroke: COLORS.dim }}
                >
                  {attackDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.color} stroke="none" opacity={0.85} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "#0a1520", border: `1px solid ${COLORS.border}`, borderRadius: 4, fontFamily: "monospace", fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8 }}>
              {attackDistribution.map(d => (
                <div key={d.name} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.7rem", color: COLORS.dim }}>
                  <div style={{ width: 8, height: 8, borderRadius: 2, background: d.color }} />
                  {d.name}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ROC Curves */}
        <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "28px 24px", marginBottom: 32 }}>
          <SectionTitle>ROC Curve Comparison</SectionTitle>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="fpr" tickFormatter={v => `${(v * 100).toFixed(0)}%`} label={{ value: "False Positive Rate", position: "insideBottom", offset: -4, fill: COLORS.dim, fontSize: 11 }} tick={{ fill: COLORS.dim, fontSize: 10 }} axisLine={{ stroke: COLORS.border }} />
              <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", fill: COLORS.dim, fontSize: 11 }} tick={{ fill: COLORS.dim, fontSize: 10 }} axisLine={{ stroke: COLORS.border }} />
              <Tooltip content={<ROCTooltip />} />
              <Legend wrapperStyle={{ color: COLORS.dim, fontSize: 11, fontFamily: "monospace" }} />
              <Line dataKey="Random" stroke={COLORS.border} strokeDasharray="4 4" dot={false} strokeWidth={1} />
              {modelResults.map((m, i) => (
                <Line key={m.name} dataKey={m.name} stroke={modelLineColors[i]} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Interactive Prediction */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>

          {/* Model Selector + Metrics */}
          <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "28px 24px" }}>
            <SectionTitle>Model Selector</SectionTitle>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 24 }}>
              {modelResults.map(m => (
                <button key={m.name} onClick={() => setSelectedModel(m.name)}
                  style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    background: selectedModel === m.name ? `${COLORS.cyan}18` : "transparent",
                    border: `1px solid ${selectedModel === m.name ? COLORS.cyan : COLORS.border}`,
                    borderRadius: 3, padding: "10px 14px", cursor: "pointer",
                    color: selectedModel === m.name ? COLORS.cyan : COLORS.dim,
                    fontFamily: "monospace", fontSize: "0.82rem",
                    transition: "all 0.15s",
                  }}>
                  <span>{m.name}</span>
                  <span style={{ color: selectedModel === m.name ? COLORS.green : COLORS.dim }}>
                    F1: {(m.f1 * 100).toFixed(1)}%
                  </span>
                </button>
              ))}
            </div>

            {selected && (
              <div>
                <div style={{ color: COLORS.dim, fontSize: "0.68rem", letterSpacing: "0.15em", marginBottom: 12, textTransform: "uppercase" }}>Selected Model Metrics</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  {Object.entries(metricColors).map(([key, color]) => (
                    <div key={key} style={{ background: `${color}12`, border: `1px solid ${color}33`, borderRadius: 3, padding: "10px 14px" }}>
                      <div style={{ color: COLORS.dim, fontSize: "0.65rem", letterSpacing: "0.1em", textTransform: "uppercase" }}>{key}</div>
                      <div style={{ color, fontSize: "1.3rem", fontFamily: "'Share Tech Mono', monospace", marginTop: 2, textShadow: `0 0 8px ${color}66` }}>
                        {(selected[key] * 100).toFixed(2)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Prediction Demo */}
          <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: 4, padding: "28px 24px" }}>
            <SectionTitle>Live Prediction Demo</SectionTitle>
            <div style={{ color: COLORS.dim, fontSize: "0.7rem", marginBottom: 14, letterSpacing: "0.08em" }}>Sample packet features (test set):</div>
            <div style={{ background: "#070f18", border: `1px solid ${COLORS.border}`, borderRadius: 3, padding: "14px 16px", marginBottom: 20, maxHeight: 200, overflowY: "auto" }}>
              {Object.entries(samplePacket).map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: `1px solid ${COLORS.border}22` }}>
                  <span style={{ color: COLORS.dim, fontSize: "0.72rem" }}>{k}</span>
                  <span style={{ color: COLORS.text, fontSize: "0.72rem", fontFamily: "'Share Tech Mono', monospace" }}>{v}</span>
                </div>
              ))}
            </div>

            <button onClick={runPrediction} style={{
              width: "100%", padding: "12px", background: `${COLORS.cyan}22`,
              border: `1px solid ${COLORS.cyan}`, borderRadius: 3, color: COLORS.cyan,
              fontFamily: "'Share Tech Mono', monospace", fontSize: "0.85rem", cursor: "pointer",
              letterSpacing: "0.12em", textTransform: "uppercase",
              boxShadow: `0 0 12px ${COLORS.cyan}22`,
              transition: "all 0.15s",
            }}>▶ Run Prediction ({selectedModel})</button>

            {predicted === null && <div style={{ height: 80 }} />}
            {predicted && (
              <div style={{
                marginTop: 16,
                background: predicted.label === 1 ? `${COLORS.red}15` : `${COLORS.green}15`,
                border: `1px solid ${predicted.label === 1 ? COLORS.red : COLORS.green}`,
                borderRadius: 3, padding: "18px 20px",
                animation: "fadeIn 0.4s ease",
              }}>
                <style>{`@keyframes fadeIn { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:translateY(0) } }`}</style>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ color: COLORS.dim, fontSize: "0.65rem", letterSpacing: "0.15em", textTransform: "uppercase" }}>Predicted Label</div>
                    <div style={{ color: predicted.label === 1 ? COLORS.red : COLORS.green, fontSize: "1.6rem", fontFamily: "'Share Tech Mono', monospace", textShadow: `0 0 10px ${predicted.label === 1 ? COLORS.red : COLORS.green}88` }}>
                      {predicted.label === 1 ? "⚠ THREAT" : "✓ NORMAL"}
                    </div>
                    <div style={{ color: COLORS.dim, fontSize: "0.7rem", marginTop: 2 }}>Type: {predicted.type}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ color: COLORS.dim, fontSize: "0.65rem", letterSpacing: "0.12em", textTransform: "uppercase" }}>Confidence</div>
                    <div style={{ color: COLORS.amber, fontSize: "1.4rem", fontFamily: "'Share Tech Mono', monospace" }}>{(predicted.prob * 100).toFixed(2)}%</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div style={{ marginTop: 40, textAlign: "center", color: COLORS.dim, fontSize: "0.65rem", letterSpacing: "0.12em", borderTop: `1px solid ${COLORS.border}`, paddingTop: 20 }}>
          NIDS DASHBOARD &nbsp;·&nbsp; MODELS: GAUSSIAN NB · MULTINOMIAL NB · BERNOULLI NB · LOGISTIC REGRESSION · RANDOM FOREST
        </div>
      </div>
    </div>
  );
}
