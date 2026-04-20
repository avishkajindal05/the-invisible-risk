import React, { useState } from 'react';
import './index.css';

const FIELD_OPTS = {
  NAME_CONTRACT_TYPE: ['Cash loans', 'Revolving loans'],
  CODE_GENDER: ['F', 'M', 'XNA'],
  FLAG_OWN_CAR: ['Y', 'N'],
  FLAG_OWN_REALTY: ['Y', 'N'],
  NAME_INCOME_TYPE: ['Working', 'State servant', 'Commercial associate', 'Pensioner', 'Unemployed', 'Student', 'Businessman'],
  NAME_EDUCATION_TYPE: ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
  NAME_FAMILY_STATUS: ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'],
  NAME_HOUSING_TYPE: ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment'],
};

const GENDER_LABELS = { F: 'Female', M: 'Male', XNA: 'Not specified' };

const defaultForm = {
  SK_ID_CURR: '',
  NAME_CONTRACT_TYPE: 'Cash loans',
  CODE_GENDER: 'F',
  FLAG_OWN_CAR: 'N',
  FLAG_OWN_REALTY: 'Y',
  CNT_CHILDREN: '0',
  AMT_INCOME_TOTAL: '',
  AMT_CREDIT: '',
  AMT_ANNUITY: '',
  AMT_GOODS_PRICE: '',
  NAME_INCOME_TYPE: 'Working',
  NAME_EDUCATION_TYPE: 'Secondary / secondary special',
  NAME_FAMILY_STATUS: 'Married',
  NAME_HOUSING_TYPE: 'House / apartment',
  DAYS_BIRTH: '',
  DAYS_EMPLOYED: '',
};

function RiskGauge({ probability }) {
  const pct = Math.round(probability * 100);
  const angle = (probability * 180) - 90; // -90 to 90
  const color = probability > 0.6 ? '#ef4444' : probability > 0.35 ? '#f59e0b' : '#10b981';
  return (
    <div className="gauge-wrap">
      <div className="gauge">
        <div className="gauge-arc" />
        <div className="gauge-needle" style={{ transform: `rotate(${angle}deg)` }} />
        <div className="gauge-center">
          <span className="gauge-pct" style={{ color }}>{pct}%</span>
          <span className="gauge-label">Default Probability</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState(defaultForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setResult(null); setError(null);

    const payload = {
      NAME_CONTRACT_TYPE: form.NAME_CONTRACT_TYPE,
      CODE_GENDER: form.CODE_GENDER,
      FLAG_OWN_CAR: form.FLAG_OWN_CAR,
      FLAG_OWN_REALTY: form.FLAG_OWN_REALTY,
      CNT_CHILDREN: parseInt(form.CNT_CHILDREN || 0, 10),
      AMT_INCOME_TOTAL: parseFloat(form.AMT_INCOME_TOTAL || 0),
      AMT_CREDIT: parseFloat(form.AMT_CREDIT || 0),
      AMT_ANNUITY: parseFloat(form.AMT_ANNUITY || 0),
      AMT_GOODS_PRICE: parseFloat(form.AMT_GOODS_PRICE || 0),
      NAME_INCOME_TYPE: form.NAME_INCOME_TYPE,
      NAME_EDUCATION_TYPE: form.NAME_EDUCATION_TYPE,
      NAME_FAMILY_STATUS: form.NAME_FAMILY_STATUS,
      NAME_HOUSING_TYPE: form.NAME_HOUSING_TYPE,
      DAYS_BIRTH: parseInt(form.DAYS_BIRTH || 0, 10),
      DAYS_EMPLOYED: parseInt(form.DAYS_EMPLOYED || 0, 10),
    };
    if (form.SK_ID_CURR) payload.SK_ID_CURR = parseInt(form.SK_ID_CURR, 10);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isHighRisk = result?.prediction === 1;

  return (
    <div className="app">
      <div className="bg-blobs">
        <div className="blob b1" /><div className="blob b2" /><div className="blob b3" />
      </div>

      <div className="shell">
        {/* ── Header ── */}
        <header className="page-header">
          <div className="header-badge">AI-POWERED</div>
          <h1>Credit Risk Assessor</h1>
          <p>Enter applicant details below. Applicant ID is optional — new applicants are fully supported.</p>
        </header>

        <div className="layout">
          {/* ── Form Panel ── */}
          <form className="card form-card" onSubmit={handleSubmit}>
            <h2 className="card-title">Applicant Information</h2>

            {/* ID — optional */}
            <div className="field-group id-group">
              <label>
                Applicant ID <span className="optional-badge">optional</span>
              </label>
              <input
                type="number"
                id="sk_id_curr"
                placeholder="e.g. 100002 — leave blank for new applicant"
                value={form.SK_ID_CURR}
                onChange={e => set('SK_ID_CURR', e.target.value)}
              />
              <p className="hint">If provided and found, the full 200-feature model is used for higher accuracy.</p>
            </div>

            <div className="divider" />

            {/* ── Section: Loan ── */}
            <h3 className="section-label">Loan Details</h3>
            <div className="grid-2">
              <div className="field-group">
                <label>Contract Type</label>
                <select value={form.NAME_CONTRACT_TYPE} onChange={e => set('NAME_CONTRACT_TYPE', e.target.value)}>
                  {FIELD_OPTS.NAME_CONTRACT_TYPE.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div className="field-group">
                <label>Credit Amount (₹)</label>
                <input type="number" required placeholder="0.00" value={form.AMT_CREDIT} onChange={e => set('AMT_CREDIT', e.target.value)} />
              </div>
              <div className="field-group">
                <label>Annuity (₹)</label>
                <input type="number" required placeholder="0.00" value={form.AMT_ANNUITY} onChange={e => set('AMT_ANNUITY', e.target.value)} />
              </div>
              <div className="field-group">
                <label>Goods Price (₹)</label>
                <input type="number" required placeholder="0.00" value={form.AMT_GOODS_PRICE} onChange={e => set('AMT_GOODS_PRICE', e.target.value)} />
              </div>
            </div>

            {/* ── Section: Personal ── */}
            <h3 className="section-label">Personal Details</h3>
            <div className="grid-2">
              <div className="field-group">
                <label>Gender</label>
                <select value={form.CODE_GENDER} onChange={e => set('CODE_GENDER', e.target.value)}>
                  {FIELD_OPTS.CODE_GENDER.map(o => <option key={o} value={o}>{GENDER_LABELS[o]}</option>)}
                </select>
              </div>
              <div className="field-group">
                <label>Annual Income (₹)</label>
                <input type="number" required placeholder="0.00" value={form.AMT_INCOME_TOTAL} onChange={e => set('AMT_INCOME_TOTAL', e.target.value)} />
              </div>
              <div className="field-group">
                <label>Children Count</label>
                <input type="number" min="0" value={form.CNT_CHILDREN} onChange={e => set('CNT_CHILDREN', e.target.value)} />
              </div>
              <div className="field-group">
                <label>Age (days before today, negative)</label>
                <input type="number" required placeholder="e.g. -12000" value={form.DAYS_BIRTH} onChange={e => set('DAYS_BIRTH', e.target.value)} />
              </div>
              <div className="field-group">
                <label>Employed since (days, negative)</label>
                <input type="number" required placeholder="e.g. -2000" value={form.DAYS_EMPLOYED} onChange={e => set('DAYS_EMPLOYED', e.target.value)} />
              </div>
              <div className="field-group">
                <label>Owns Car</label>
                <select value={form.FLAG_OWN_CAR} onChange={e => set('FLAG_OWN_CAR', e.target.value)}>
                  {FIELD_OPTS.FLAG_OWN_CAR.map(o => <option key={o} value={o}>{o === 'Y' ? 'Yes' : 'No'}</option>)}
                </select>
              </div>
              <div className="field-group">
                <label>Owns Realty</label>
                <select value={form.FLAG_OWN_REALTY} onChange={e => set('FLAG_OWN_REALTY', e.target.value)}>
                  {FIELD_OPTS.FLAG_OWN_REALTY.map(o => <option key={o} value={o}>{o === 'Y' ? 'Yes' : 'No'}</option>)}
                </select>
              </div>
            </div>

            {/* ── Section: Background ── */}
            <h3 className="section-label">Background</h3>
            <div className="grid-2">
              <div className="field-group">
                <label>Income Type</label>
                <select value={form.NAME_INCOME_TYPE} onChange={e => set('NAME_INCOME_TYPE', e.target.value)}>
                  {FIELD_OPTS.NAME_INCOME_TYPE.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div className="field-group">
                <label>Education</label>
                <select value={form.NAME_EDUCATION_TYPE} onChange={e => set('NAME_EDUCATION_TYPE', e.target.value)}>
                  {FIELD_OPTS.NAME_EDUCATION_TYPE.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div className="field-group">
                <label>Family Status</label>
                <select value={form.NAME_FAMILY_STATUS} onChange={e => set('NAME_FAMILY_STATUS', e.target.value)}>
                  {FIELD_OPTS.NAME_FAMILY_STATUS.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div className="field-group">
                <label>Housing Type</label>
                <select value={form.NAME_HOUSING_TYPE} onChange={e => set('NAME_HOUSING_TYPE', e.target.value)}>
                  {FIELD_OPTS.NAME_HOUSING_TYPE.map(o => <option key={o}>{o}</option>)}
                </select>
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading} id="submit-btn">
              {loading
                ? <><span className="spinner" /> Analysing…</>
                : 'Assess Credit Risk'}
            </button>
          </form>

          {/* ── Result Panel ── */}
          <aside className="card result-card">
            {!result && !error && !loading && (
              <div className="empty-state">
                <div className="empty-icon">🔮</div>
                <p>Your AI risk assessment will appear here once you submit the form.</p>
              </div>
            )}
            {loading && (
              <div className="empty-state">
                <div className="pulsing-ring" />
                <p>Running AI analysis…</p>
              </div>
            )}
            {error && (
              <div className="alert alert-error">
                <span>⚠</span> {error}
              </div>
            )}
            {result && (
              <div className={`result-body fade-in ${isHighRisk ? 'high-risk' : 'low-risk'}`}>
                <div className={`verdict-badge ${isHighRisk ? 'danger' : 'safe'}`}>
                  {isHighRisk ? '⛔ HIGH RISK' : '✅ LOW RISK'}
                </div>

                <RiskGauge probability={result.probability} />

                <div className="result-rows">
                  <div className="result-row">
                    <span>Default Probability</span>
                    <strong>{(result.probability * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="result-row">
                    <span>Decision Threshold</span>
                    <strong>{(result.threshold * 100).toFixed(0)}%</strong>
                  </div>
                  <div className="result-row">
                    <span>History Found</span>
                    <strong>{result.has_history ? 'Yes' : 'No (new applicant)'}</strong>
                  </div>
                </div>

                <div className="model-tag">{result.model_used}</div>

                <p className={`verdict-text ${isHighRisk ? 'text-danger' : 'text-success'}`}>
                  {isHighRisk
                    ? 'This applicant exceeds the risk tolerance threshold and is likely to default.'
                    : 'This applicant meets the safety criteria and shows a low risk of default.'}
                </p>
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
