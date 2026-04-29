import { useState, useEffect, useMemo } from "react";

// Use environment variable for production, fallback to local for development
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const numericFields = [
  { key: "T2M_MAX", label: "درجة الحرارة العظمى (°C)", placeholder: "35.2", min: -40, max: 60 },
  { key: "T2M_MIN", label: "درجة الحرارة الصغرى (°C)", placeholder: "18.5", min: -40, max: 60 },
  { key: "RH2M", label: "الرطوبة النسبية (%)", placeholder: "45.0", min: 0, max: 100 },
  { key: "WS2M", label: "سرعة الرياح (m/s)", placeholder: "3.2", min: 0, max: 15 },
  { key: "PS", label: "الضغط الجوي (kPa)", placeholder: "97.5", min: 50, max: 110 },
  { key: "Rn", label: "الإشعاع الصافي Rn (MJ)", placeholder: "7.3", min: -5, max: 30 },
  { key: "FieldCapacity", label: "سعة الحقل (FC)", placeholder: "0.35", min: 0, max: 1, step: "0.01" },
  { key: "WiltingPoint", label: "نقطة الذبول (WP)", placeholder: "0.15", min: 0, max: 1, step: "0.01" },
  { key: "RootDepth", label: "عمق الجذور (m)", placeholder: "0.5", min: 0, max: 3, step: "0.01" },
  { key: "Kc", label: "معامل المحصول (Kc)", placeholder: "0.85", min: 0.1, max: 2.0, step: "0.01" },
];

const KC_MAP = {
  "Seedling": 0.7,
  "Vegetative": 1.0,
  "Flowering": 1.05,
  "Maturity": 0.95,
  "Initial": 0.7,
  "Mid-season": 1.05,
  "Late-season": 0.95,
  "Harvest": 0.90,
};

export default function PredictPage() {
  const [classes, setClasses] = useState({ CROP: [], GrowthStage: [], SOIL_TEX: [] });
  const [form, setForm] = useState({ Kc: "0.85" });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const tMean = useMemo(() => {
    const max = parseFloat(form.T2M_MAX);
    const min = parseFloat(form.T2M_MIN);
    if (isNaN(max) || isNaN(min)) return "";
    return ((max + min) / 2).toFixed(2);
  }, [form.T2M_MAX, form.T2M_MIN]);

  useEffect(() => {
    fetch(`${API_URL}/api/classes`)
      .then((r) => r.json())
      .then((data) => {
        setClasses(data);
        const firstStage = data.GrowthStage[0] || "";
        setForm((f) => ({
          ...f,
          CROP: data.CROP[0] || "",
          GrowthStage: firstStage,
          SOIL_TEX: data.SOIL_TEX[0] || "",
          Kc: KC_MAP[firstStage] || "0.85"
        }));
      })
      .catch(() => setError("تعذر الاتصال بالـ API — تأكد إن api_v2.py شغال على port 5000"));
  }, []);

  // Auto-update Kc when GrowthStage changes
  useEffect(() => {
    if (form.GrowthStage && KC_MAP[form.GrowthStage]) {
      setForm(f => ({ ...f, Kc: KC_MAP[form.GrowthStage].toString() }));
    }
  }, [form.GrowthStage]);

  const handleChange = (key, value) => {
    setForm((f) => ({ ...f, [key]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    const payload = { ...form, T_mean: tMean, G: 0.0 };

    try {
      const res = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 16px", fontFamily: "'Audiowide', sans-serif" }}>
      <h2 style={{ fontSize: 32, textAlign: "center", marginBottom: 32, color: "var(--color-text)", textShadow: "0 0 12px rgba(59, 130, 246, 0.4)" }}>
        🔮 احسب التنبؤ بكمية المياه والبخر نتح
      </h2>

      <form onSubmit={handleSubmit}>
        {/* ── Temperature Section ─────────────────────────────────────── */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
          <div style={{ fontSize: 20, color: "var(--color-primary)", marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🌡️</span> درجات الحرارة
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 24 }}>
            {numericFields.filter(f => ["T2M_MAX", "T2M_MIN"].includes(f.key)).map(({ key, label, placeholder, min, max }) => (
              <div key={key}>
                <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>{label}</label>
                <input
                  type="number"
                  step="any"
                  required
                  placeholder={placeholder}
                  value={form[key] || ""}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.6)", color: "var(--color-text)", fontFamily: "'JetBrains Mono', monospace", fontSize: 16, borderRadius: 4, outline: "none", transition: "all 0.2s" }}
                  min={min}
                  max={max}
                  onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; e.target.style.boxShadow = "0 0 8px rgba(59, 130, 246, 0.4)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--color-border)"; e.target.style.boxShadow = "none"; }}
                />
              </div>
            ))}

            {/* T_mean — auto-calculated */}
            <div>
              <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>متوسط درجة الحرارة (محسوب)</label>
              <input
                type="text"
                readOnly
                value={tMean || "—"}
                style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.3)", color: "var(--color-text-muted)", fontFamily: "'JetBrains Mono', monospace", fontSize: 16, borderRadius: 4, cursor: "not-allowed" }}
              />
            </div>
          </div>
        </div>

        {/* ── Weather & Radiation Section ─────────────────────────────── */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
          <div style={{ fontSize: 20, color: "var(--color-primary)", marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🌤️</span> الأرصاد الجوية والإشعاع
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 24 }}>
            {numericFields.filter(f => ["RH2M", "WS2M", "PS", "Rn"].includes(f.key)).map(({ key, label, placeholder, min, max, step }) => (
              <div key={key}>
                <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>{label}</label>
                <input
                  type="number"
                  step={step || "any"}
                  required
                  placeholder={placeholder}
                  value={form[key] || ""}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.6)", color: "var(--color-text)", fontFamily: "'JetBrains Mono', monospace", fontSize: 16, borderRadius: 4, outline: "none", transition: "all 0.2s" }}
                  min={min}
                  max={max}
                  onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; e.target.style.boxShadow = "0 0 8px rgba(59, 130, 246, 0.4)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--color-border)"; e.target.style.boxShadow = "none"; }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* ── Soil Properties Section ─────────────────────────────────────── */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
          <div style={{ fontSize: 20, color: "var(--color-primary)", marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🧱</span> خصائص التربة
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 24 }}>
            {numericFields.filter(f => ["FieldCapacity", "WiltingPoint"].includes(f.key)).map(({ key, label, placeholder, min, max, step }) => (
              <div key={key}>
                <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>{label}</label>
                <input
                  type="number"
                  step={step || "any"}
                  required
                  placeholder={placeholder}
                  value={form[key] || ""}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.6)", color: "var(--color-text)", fontFamily: "'JetBrains Mono', monospace", fontSize: 16, borderRadius: 4, outline: "none", transition: "all 0.2s" }}
                  min={min}
                  max={max}
                  onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; e.target.style.boxShadow = "0 0 8px rgba(59, 130, 246, 0.4)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--color-border)"; e.target.style.boxShadow = "none"; }}
                />
              </div>
            ))}

            <div key="SOIL_TEX">
              <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>نوع التربة</label>
              <select
                value={form.SOIL_TEX || ""}
                onChange={(e) => handleChange("SOIL_TEX", e.target.value)}
                style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.6)", color: "var(--color-text)", fontFamily: "'Audiowide', sans-serif", fontSize: 16, borderRadius: 4, outline: "none", cursor: "pointer" }}
              >
                {(classes.SOIL_TEX || []).map((o) => <option key={o} value={o} style={{ background: "var(--color-surface-card)", color: "var(--color-text)" }}>{o}</option>)}
              </select>
            </div>
          </div>
        </div>

        {/* ── Crop Data Section ─────────────────────────────────────── */}
        <div style={{ background: "var(--color-surface-card)", borderRadius: 8, padding: 24, marginBottom: 32, border: "1px solid var(--color-border)", boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
          <div style={{ fontSize: 20, color: "var(--color-primary)", marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🌿</span> بيانات المحصول
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 24 }}>
            {[
              { key: "CROP", label: "نوع المحصول", options: classes.CROP },
              { key: "GrowthStage", label: "مرحلة النمو", options: classes.GrowthStage },
            ].map(({ key, label, options }) => (
              <div key={key}>
                <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>{label}</label>
                <select
                  value={form[key] || ""}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.6)", color: "var(--color-text)", fontFamily: "'Audiowide', sans-serif", fontSize: 16, borderRadius: 4, outline: "none", cursor: "pointer" }}
                >
                  {(options || []).map((o) => <option key={o} value={o} style={{ background: "var(--color-surface-card)", color: "var(--color-text)" }}>{o}</option>)}
                </select>
              </div>
            ))}

            {numericFields.filter(f => ["RootDepth", "Kc"].includes(f.key)).map(({ key, label, placeholder, min, max, step }) => (
              <div key={key}>
                <label style={{ display: "block", fontSize: 14, marginBottom: 8, color: "var(--color-text-muted)" }}>{label}</label>
                <input
                  type="number"
                  step={step || "any"}
                  required
                  placeholder={placeholder}
                  value={form[key] || ""}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={{ width: "100%", boxSizing: "border-box", padding: "12px", border: "1px solid var(--color-border)", background: "rgba(15, 23, 42, 0.6)", color: "var(--color-text)", fontFamily: "'JetBrains Mono', monospace", fontSize: 16, borderRadius: 4, outline: "none", transition: "all 0.2s" }}
                  min={min}
                  max={max}
                  onFocus={(e) => { e.target.style.borderColor = "var(--color-primary)"; e.target.style.boxShadow = "0 0 8px rgba(59, 130, 246, 0.4)"; }}
                  onBlur={(e) => { e.target.style.borderColor = "var(--color-border)"; e.target.style.boxShadow = "none"; }}
                />
              </div>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{ width: "100%", padding: "16px", fontSize: "20px", fontFamily: "'Audiowide', sans-serif", background: "var(--color-primary)", color: "#FFFFFF", border: "none", borderRadius: 4, cursor: "pointer", transition: "all 0.2s ease", marginTop: "16px", boxShadow: loading ? "none" : "0 0 12px rgba(59, 130, 246, 0.6)", opacity: loading ? 0.5 : 1 }}
          onMouseEnter={(e) => { if (!loading) e.target.style.filter = "brightness(1.1)"; }}
          onMouseLeave={(e) => { if (!loading) e.target.style.filter = "none"; }}
        >
          {loading ? "جاري الحساب..." : "احسب الآن"}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div style={{ background: "rgba(220, 38, 38, 0.1)", color: "var(--color-danger)", padding: "16px", border: "1px solid var(--color-danger)", borderRadius: 4, margin: "24px 0", display: "flex", alignItems: "center", gap: 12, boxShadow: "0 0 8px rgba(220, 38, 38, 0.4)" }} role="alert">
          <span style={{ fontSize: 24 }}>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* Result */}
      {result && (
        <div style={{ marginTop: 48, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 24 }}>
          {/* ETc Card */}
          <div style={{ background: "var(--color-surface-card)", border: "1px solid var(--color-secondary)", borderRadius: 8, padding: "32px 24px", textAlign: "center", boxShadow: "0 0 24px rgba(139, 92, 246, 0.2)" }}>
            <div style={{ fontSize: 16, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 12 }}>البخر نتح الفعلي (ETc)</div>
            <div style={{ fontSize: 48, color: "var(--color-secondary)", fontFamily: "'JetBrains Mono', monospace", margin: "16px 0", textShadow: "0 0 12px rgba(139, 92, 246, 0.6)" }}>{result.ETc}</div>
            <div style={{ display: "inline-block", fontSize: 12, background: "rgba(15, 23, 42, 0.8)", color: "var(--color-text-muted)", padding: "4px 12px", border: "1px solid var(--color-border)", borderRadius: 12 }}>mm/day</div>
          </div>

          {/* Water Amount Card */}
          <div style={{ background: "var(--color-surface-card)", border: "1px solid var(--color-success)", borderRadius: 8, padding: "32px 24px", textAlign: "center", boxShadow: "0 0 24px rgba(22, 163, 74, 0.2)" }}>
            <div style={{ fontSize: 16, color: "var(--color-text-muted)", textTransform: "uppercase", marginBottom: 12 }}>كمية المياه الواجب إضافتها</div>
            <div style={{ fontSize: 48, color: "var(--color-success)", fontFamily: "'JetBrains Mono', monospace", margin: "16px 0", textShadow: "0 0 12px rgba(22, 163, 74, 0.6)" }}>{result.Water}</div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 12 }}>
              <span style={{ display: "inline-block", fontSize: 12, background: "rgba(15, 23, 42, 0.8)", color: "var(--color-text-muted)", padding: "4px 12px", border: "1px solid var(--color-border)", borderRadius: 12 }}>m³/acre</span>
              <span style={{ display: "inline-block", fontSize: 12, background: "rgba(15, 23, 42, 0.8)", color: "var(--color-warning)", padding: "4px 12px", border: "1px solid var(--color-border)", borderRadius: 12, borderColor: 'var(--color-warning)' }}>
                {result.water_source === "ml_model" ? "ML Model" : "Fallback"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
