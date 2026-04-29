import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

const results = {
  Etc: {
    "XGBoost (v2)": { MAE: 0.0351, RMSE: 0.0459, R2: 0.9997, MAPE: 0.71 },
    "Random Forest (v1)": { MAE: 0.0639, RMSE: 0.1053, R2: 0.9999, MAPE: 0.26 },
    "Gradient Boosting": { MAE: 0.4638, RMSE: 0.6262, R2: 0.9966, MAPE: 1.93 },
    "KNN (k=3)": { MAE: 0.6816, RMSE: 0.9264, R2: 0.9926, MAPE: 3.0 },
    "Lasso Regression": { MAE: 1.245, RMSE: 1.6113, R2: 0.9776, MAPE: 5.89 },
    "Linear Regression": { MAE: 1.2083, RMSE: 1.5985, R2: 0.9779, MAPE: 5.94 },
  },
};

const featureImportance = {
  Etc: { T2M_MAX: 0.4, "GrowthStage_Seedling": 0.1999, Rn: 0.1419, T_mean: 0.0877, "GrowthStage_Flowering": 0.0596, "GrowthStage_Vegetative": 0.0348, "GrowthStage_Maturity": 0.0292, WS2M: 0.0246, RH2M: 0.0187, T2M_MIN: 0.0019, PS: 0.0018 },
};

const featureLabels = {
  T2M_MAX: "درجة الحرارة العظمى",
  T2M_MIN: "درجة الحرارة الصغرى",
  T_mean: "متوسط الحرارة",
  RH2M: "الرطوبة النسبية",
  WS2M: "سرعة الرياح",
  PS: "الضغط الجوي",
  Rn: "الإشعاع الصافي",
  "GrowthStage_Seedling": "مرحلة البادرة",
  "GrowthStage_Vegetative": "مرحلة النمو الخضري",
  "GrowthStage_Flowering": "مرحلة الإزهار",
  "GrowthStage_Maturity": "مرحلة النضج",
};

const targetLabels = {
  Etc: "البخر نتح الفعلي (ETc)",
};

const COLORS = ["#3B82F6", "#8B5CF6", "#16A34A", "#D97706", "#DC2626", "#06b6d4", "#f43f5e"];

export default function Dashboard() {
  const [activeTarget, setActiveTarget] = useState("Etc");
  const [activeMetric, setActiveMetric] = useState("R2");

  const targetData = results[activeTarget];
  const modelNames = Object.keys(targetData);

  const comparisonData = modelNames.map((name, i) => ({
    name: name.length > 12 ? name.substring(0, 12) + "…" : name,
    fullName: name,
    ...targetData[name],
  }));

  const fiData = Object.entries(featureImportance[activeTarget])
    .filter(([_, v]) => v > 0)
    .sort((a, b) => b[1] - a[1])
    .map(([key, val]) => ({
      feature: featureLabels[key] || key,
      importance: val,
      pct: (val * 100).toFixed(1),
    }));

  const bestModel = modelNames.reduce((best, name) =>
    targetData[name].R2 > targetData[best].R2 ? name : best, modelNames[0]
  );

  const metricLabels = { R2: "R² Score", MAE: "MAE", RMSE: "RMSE", MAPE: "MAPE %" };

  return (
    <div>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 16px", fontFamily: "'Audiowide', sans-serif" }}>
        {/* Target Selector */}
        <div style={{ display: "flex", gap: 12, justifyContent: "center", marginBottom: 32, flexWrap: "wrap" }}>
          {Object.keys(results).map((t) => (
            <button
              key={t}
              onClick={() => setActiveTarget(t)}
              style={{
                padding: "12px 24px",
                borderRadius: 4,
                border: "1px solid var(--color-border)",
                background: activeTarget === t ? "var(--color-primary)" : "rgba(15, 23, 42, 0.6)",
                color: activeTarget === t ? "#FFFFFF" : "var(--color-text-muted)",
                cursor: "pointer",
                fontSize: 16,
                fontFamily: "'Audiowide', sans-serif",
                boxShadow: activeTarget === t ? "0 0 12px rgba(59, 130, 246, 0.6)" : "none",
                transition: "all 0.2s ease",
              }}
            >
              {targetLabels[t]}
            </button>
          ))}
        </div>

        {/* Best Model Card */}
        <div style={{
          background: "var(--color-surface-card)",
          borderRadius: 8,
          padding: "24px",
          marginBottom: 32,
          border: "1px solid var(--color-success)",
          boxShadow: "0 0 24px rgba(22, 163, 74, 0.2)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 16,
        }}>
          <div>
            <div style={{ fontSize: 12, color: "var(--color-success)", letterSpacing: 1, textTransform: "uppercase" }}>أفضل نموذج / Best Model</div>
            <div style={{ fontSize: 24, color: "var(--color-text)", marginTop: 4, textShadow: "0 0 8px rgba(22, 163, 74, 0.4)" }}>{bestModel}</div>
          </div>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            {Object.entries(targetData[bestModel]).map(([k, v]) => (
              <div key={k} style={{ textAlign: "center" }}>
                <div style={{ fontSize: 12, color: "var(--color-text-muted)" }}>{k}</div>
                <div style={{ fontSize: 24, color: "#FFFFFF", fontFamily: "'JetBrains Mono', monospace" }}>{typeof v === "number" ? (k === "MAPE" ? v + "%" : v.toFixed(4)) : v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Metric Selector */}
        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          {Object.entries(metricLabels).map(([k, label]) => (
            <button
              key={k}
              onClick={() => setActiveMetric(k)}
              style={{
                padding: "8px 16px",
                borderRadius: 4,
                border: "1px solid var(--color-border)",
                background: activeMetric === k ? "var(--color-secondary)" : "rgba(15, 23, 42, 0.6)",
                color: activeMetric === k ? "#FFFFFF" : "var(--color-text-muted)",
                cursor: "pointer",
                fontSize: 14,
                fontFamily: "'Audiowide', sans-serif",
                boxShadow: activeMetric === k ? "0 0 12px rgba(139, 92, 246, 0.6)" : "none",
                transition: "all 0.2s ease",
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Model Comparison Chart */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)" }}>
          <h2 style={{ fontSize: 20, margin: "0 0 24px", color: "var(--color-text)", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "var(--color-primary)" }}>📊</span> مقارنة الموديلات — {metricLabels[activeMetric]}
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData} margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis dataKey="name" tick={{ fill: "var(--color-text-muted)", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} />
              <YAxis 
                width={60}
                tick={{ fill: "var(--color-text-muted)", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} 
                tickMargin={15}
              />
              <Tooltip
                contentStyle={{ background: "var(--color-surface-card)", border: "1px solid var(--color-border)", borderRadius: 4, color: "var(--color-text)", fontFamily: "'JetBrains Mono', monospace" }}
                formatter={(val, name, props) => [
                  typeof val === "number" ? val.toFixed(4) : val,
                  props.payload.fullName,
                ]}
              />
              <Bar dataKey={activeMetric} radius={[4, 4, 0, 0]}>
                {comparisonData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* All Models Table */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", overflowX: "auto", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)" }}>
          <h2 style={{ fontSize: 20, margin: "0 0 24px", color: "var(--color-text)", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "var(--color-primary)" }}>📋</span> جدول النتائج الكاملة
          </h2>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14, fontFamily: "'JetBrains Mono', monospace" }}>
            <thead>
              <tr>
                {["النموذج / Model", "R²", "RMSE", "MAE", "MAPE %"].map((h) => (
                  <th key={h} style={{ padding: "12px", textAlign: "center", borderBottom: "2px solid var(--color-primary)", color: "var(--color-primary)", fontFamily: "'Audiowide', sans-serif" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {modelNames.map((name, i) => {
                const m = targetData[name];
                const isBest = name === bestModel;
                return (
                  <tr key={name} style={{ background: isBest ? "rgba(59, 130, 246, 0.1)" : "transparent" }}>
                    <td style={{ padding: "12px", color: isBest ? "var(--color-primary)" : "var(--color-text)", fontFamily: "'Audiowide', sans-serif", textShadow: isBest ? "0 0 8px rgba(59, 130, 246, 0.4)" : "none" }}>
                      {isBest && "⭐ "}{name}
                    </td>
                    <td style={{ padding: "12px", textAlign: "center", color: m.R2 > 0.9 ? "var(--color-success)" : m.R2 > 0.5 ? "var(--color-warning)" : "var(--color-danger)" }}>{m.R2.toFixed(4)}</td>
                    <td style={{ padding: "12px", textAlign: "center", color: "var(--color-text-muted)" }}>{m.RMSE.toFixed(4)}</td>
                    <td style={{ padding: "12px", textAlign: "center", color: "var(--color-text-muted)" }}>{m.MAE.toFixed(4)}</td>
                    <td style={{ padding: "12px", textAlign: "center", color: "var(--color-text-muted)" }}>{m.MAPE}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Feature Importance */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)" }}>
          <h2 style={{ fontSize: 20, margin: "0 0 24px", color: "var(--color-text)", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "var(--color-secondary)" }}>🎯</span> أهمية المتغيرات — Feature Importance (XGBoost)
          </h2>
          <ResponsiveContainer width="100%" height={350}>
            <BarChart data={fiData} layout="vertical" margin={{ top: 5, right: 30, left: 30, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis type="number" tick={{ fill: "var(--color-text-muted)", fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }} />
              <YAxis dataKey="feature" type="category" width={180} interval={0} tick={{ fill: "var(--color-text-muted)", fontSize: 12, textAnchor: "start", fontFamily: "'Audiowide', sans-serif" }} />
              <Tooltip
                contentStyle={{ background: "var(--color-surface-card)", border: "1px solid var(--color-border)", borderRadius: 4, color: "var(--color-text)", fontFamily: "'JetBrains Mono', monospace" }}
                formatter={(val) => [(val * 100).toFixed(1) + "%", "Importance"]}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {fiData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "var(--color-primary)" : i === 1 ? "var(--color-secondary)" : i === 2 ? "var(--color-success)" : "var(--color-warning)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Notes */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)" }}>
          <h2 style={{ fontSize: 20, margin: "0 0 16px", color: "var(--color-text)", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "var(--color-primary)" }}>📝</span> ملاحظات
          </h2>
          <div style={{ fontSize: 14, lineHeight: 1.8, color: "var(--color-text-muted)" }}>
            <p>• تم استخدام <strong style={{ color: "var(--color-text)" }}>5-Fold Cross Validation</strong> على الداتا الكاملة (153,420 صف)</p>
            <p>• <strong style={{ color: "var(--color-success)" }}>XGBoost (v2)</strong> حقق أفضل أداء بـ R² = 0.9997 وMAE = 0.035 — بدون data leakage</p>
            <p>• أهم المتغيرات المؤثرة: الحرارة العظمى (40%)، مرحلة النمو (20%)، الإشعاع الصافي (14%)</p>
            <p>• أهم المتغيرات الثابتة: نوع المحصول (3.5%)، نوع التربة (0.8%)</p>
            <p>• النموذج الجديد يستخدم مدخلات بيئية حقيقية فقط — بدون ETo أو Kc أو أي مخرجات محسوبة</p>
          </div>
        </div>
      </div>
    </div>
  );
}