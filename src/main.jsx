import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import Dashboard from "../dashboard.jsx";
import PredictPage from "./PredictPage.jsx";

const NAV_STYLE = (active) => ({
  padding: "12px 24px",
  border: "1px solid var(--color-border)",
  background: active ? "var(--color-primary)" : "rgba(15, 23, 42, 0.6)",
  color: active ? "#FFFFFF" : "var(--color-text)",
  cursor: "pointer",
  fontSize: 16,
  fontFamily: "'Audiowide', sans-serif",
  textTransform: "uppercase",
  boxShadow: active ? "0 0 12px rgba(59, 130, 246, 0.6)" : "none",
  borderRadius: "4px",
  transition: "all 0.2s ease",
});

function App() {
  const [page, setPage] = useState("dashboard");

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Audiowide&family=JetBrains+Mono:wght@400;600&display=swap');
        
        :root {
          --color-primary: #3B82F6;
          --color-secondary: #8B5CF6;
          --color-success: #16A34A;
          --color-warning: #D97706;
          --color-danger: #DC2626;
          --color-surface: #0F172A;
          --color-surface-card: #1E293B;
          --color-text: #F9FAFB;
          --color-text-muted: #9CA3AF;
          --color-border: #334155;
        }

        body {
          margin: 0;
          background-color: var(--color-surface);
          color: var(--color-text);
          font-family: 'Audiowide', sans-serif;
        }
      `}</style>
      <div style={{ background: "var(--color-surface)", minHeight: "100vh", color: "var(--color-text)", fontFamily: "'Audiowide', sans-serif" }}>
        {/* Header */}
        <div style={{ background: "var(--color-surface-card)", padding: "32px 24px", borderBottom: "1px solid var(--color-border)", textAlign: "center", boxShadow: "0 4px 24px rgba(0,0,0,0.4)" }}>
          <h1 style={{ fontSize: 32, fontWeight: 400, margin: "0 0 24px", color: "var(--color-text)", textShadow: "0 0 12px rgba(59, 130, 246, 0.5)" }}>
            🌾 نموذج التعلم الآلي للاحتياجات المائية
          </h1>
          <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
            <button style={NAV_STYLE(page === "dashboard")} onClick={() => setPage("dashboard")}>
              📊 لوحة الأداء
            </button>
            <button style={NAV_STYLE(page === "predict")} onClick={() => setPage("predict")}>
              🔮 التنبؤ
            </button>
          </div>
        </div>

        {page === "dashboard" ? <Dashboard /> : <PredictPage />}

      {/* Cosmic Footer */}
      <footer style={{
        marginTop: 64,
        padding: "32px 16px",
        borderTop: "1px solid var(--color-border)",
        background: "rgba(15, 23, 42, 0.3)",
        textAlign: "center",
        backdropFilter: "blur(10px)"
      }}>
        <div style={{
          fontSize: 14,
          color: "var(--color-text-muted)",
          fontFamily: "'Audiowide', sans-serif",
          letterSpacing: 1
        }}>
          © {new Date().getFullYear()} COSMIC IRRIGATION SYSTEM | ALL RIGHTS RESERVED
        </div>
        <div style={{
          marginTop: 8,
          fontSize: 16,
          color: "var(--color-primary)",
          fontFamily: "'Audiowide', sans-serif",
          textShadow: "0 0 8px rgba(59, 130, 246, 0.4)"
        }}>
          Developed by <span style={{ color: "var(--color-secondary)", textShadow: "0 0 8px rgba(139, 92, 246, 0.4)" }}>Marwan _ewis</span>
        </div>
      </footer>
    </div>
    </>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
