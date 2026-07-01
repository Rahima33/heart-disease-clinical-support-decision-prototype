import { Activity, AlertTriangle, ArrowRight, BarChart3, CheckCircle2, HeartPulse, Loader2, RotateCcw, Send, ShieldCheck, TrendingDown, TrendingUp } from 'lucide-react';
import { useMemo, useState } from 'react';
import heroImage from './assets/cardiorisk-hero.png';

const API_URL = import.meta.env.VITE_API_URL || '/api';

const initialForm = {
  age: 29,
  sex: 'Male',
  cp: 'typical angina',
  trestbps: 100,
  chol: 190,
  fbs: true,
  restecg: 'normal',
  thalch: 100,
  exang: false,
  oldpeak: 0,
  slope: 'upsloping',
  ca: 0,
  thal: 'normal',
  has_missing: false,
};

const fieldGroups = [
  {
    title: 'Patient',
    fields: [
      { key: 'age', label: 'Age', type: 'number', min: 1, max: 120 },
      { key: 'sex', label: 'Sex', type: 'select', options: ['Male', 'Female'] },
    ],
  },
  {
    title: 'Vitals',
    fields: [
      { key: 'trestbps', label: 'Resting BP', type: 'number', min: 0, max: 260, suffix: 'mm Hg' },
      { key: 'chol', label: 'Cholesterol', type: 'number', min: 0, max: 700, suffix: 'mg/dL' },
      { key: 'fbs', label: 'Fasting blood sugar elevated', type: 'checkbox' },
    ],
  },
  {
    title: 'Symptoms',
    fields: [
      { key: 'cp', label: 'Chest pain type', type: 'select', options: ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'] },
      { key: 'exang', label: 'Exercise-induced angina', type: 'checkbox' },
      { key: 'thalch', label: 'Max heart rate', type: 'number', min: 0, max: 240, suffix: 'bpm' },
    ],
  },
  {
    title: 'Testing',
    fields: [
      { key: 'restecg', label: 'Resting ECG', type: 'select', options: ['normal', 'lv hypertrophy', 'st-t abnormality'] },
      { key: 'oldpeak', label: 'ST depression', type: 'number', min: -5, max: 10, step: 0.1 },
      { key: 'slope', label: 'ST slope', type: 'select', options: ['upsloping', 'flat', 'downsloping'] },
      { key: 'ca', label: 'Major vessels', type: 'number', min: 0, max: 4, step: 1 },
      { key: 'thal', label: 'Thallium result', type: 'select', options: ['normal', 'fixed defect', 'reversable defect'] },
      { key: 'has_missing', label: 'Incomplete diagnostic testing', type: 'checkbox' },
    ],
  },
];

const examples = [
  {
    name: 'Current Test',
    values: initialForm,
  },
  {
    name: 'Lower Signal',
    values: { ...initialForm, age: 34, sex: 'Female', fbs: false, thalch: 178, trestbps: 112, chol: 165 },
  },
  {
    name: 'Higher Signal',
    values: { ...initialForm, age: 62, cp: 'asymptomatic', trestbps: 155, chol: 285, thalch: 112, exang: true, oldpeak: 2.3, slope: 'flat', ca: 2, thal: 'fixed defect' },
  },
];


function toPayload(form) {
  return {
    ...form,
    age: Number(form.age),
    trestbps: Number(form.trestbps),
    chol: Number(form.chol),
    thalch: Number(form.thalch),
    oldpeak: Number(form.oldpeak),
    ca: Number(form.ca),
  };
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function riskLabel(probability) {
  if (probability >= 0.7) return 'High model signal';
  if (probability >= 0.4) return 'Moderate model signal';
  return 'Lower model signal';
}

function LandingPage({ onStart }) {
  return (
    <main className="app-shell landing-shell">
      <section className="landing-hero" style={{ '--hero-image': `url(${heroImage})` }}>
        <nav className="landing-nav" aria-label="CardioRisk">
          <div className="brand-mark">
            <HeartPulse size={21} aria-hidden="true" />
            <span>CardioRisk</span>
          </div>
          <div className="api-status landing-status">
            <CheckCircle2 size={18} aria-hidden="true" />
            <span>Prediction Engine Ready</span>
          </div>
        </nav>

        <div className="landing-content">
          <div className="landing-copy">
            <p className="eyebrow">Explainable AI for Cardiology</p>
            <h1>Heart disease risk assessment with explainable AI.</h1>
            <p className="landing-summary">
              Estimate heart disease risk using a machine learning model trained on the UCI Heart Disease dataset. Explore explainable AI insights through SHAP feature importance and receive an AI-generated clinical summary for each assessment.
            </p>
            <div className="landing-actions">
              <button className="hero-button" type="button" onClick={onStart}>
                Start Assessment
                <ArrowRight size={19} aria-hidden="true" />
              </button>
              <div className="confidence-note">
                <ShieldCheck size={18} aria-hidden="true" />
                <span>Demo model for learning, not diagnosis</span>
              </div>
            </div>
          </div>

          <div className="hero-visual-panel" aria-hidden="true">
            <div className="workflow-preview">
              <div className="workflow-header">
                <HeartPulse size={34} aria-hidden="true" />
                <div>
                  <span>Assessment Workflow</span>
                  <strong>From clinical inputs to explained output</strong>
                </div>
              </div>
              <div className="workflow-steps">
                <div><HeartPulse size={18} aria-hidden="true" /><span>Patient Information</span></div>
                <div><Activity size={18} aria-hidden="true" /><span>ML Risk Prediction</span></div>
                <div><BarChart3 size={18} aria-hidden="true" /><span>SHAP Explanation</span></div>
                <div><ShieldCheck size={18} aria-hidden="true" /><span>Clinical Summary</span></div>
              </div>
              <div className="workflow-note">Educational demonstration</div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

function Field({ field, value, onChange }) {
  const inputId = `field-${field.key}`;

  if (field.type === 'checkbox') {
    return (
      <label className="check-field" htmlFor={inputId}>
        <input
          id={inputId}
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => onChange(field.key, event.target.checked)}
        />
        <span>{field.label}</span>
      </label>
    );
  }

  return (
    <label className="field" htmlFor={inputId}>
      <span>{field.label}</span>
      <div className="input-shell">
        {field.type === 'select' ? (
          <select id={inputId} value={value} onChange={(event) => onChange(field.key, event.target.value)}>
            {field.options.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        ) : (
          <input
            id={inputId}
            type="number"
            min={field.min}
            max={field.max}
            step={field.step || 1}
            value={value}
            onChange={(event) => onChange(field.key, event.target.value)}
          />
        )}
        {field.suffix && <span className="suffix">{field.suffix}</span>}
      </div>
    </label>
  );
}

function Factor({ factor }) {
  const increased = factor.shap_impact > 0;
  const Icon = increased ? TrendingUp : TrendingDown;

  return (
    <li className="factor-row">
      <div className={`factor-icon ${increased ? 'up' : 'down'}`}>
        <Icon size={18} aria-hidden="true" />
      </div>
      <div>
        <strong>{factor.display_feature}</strong>
        <span>{increased ? 'Increased predicted risk' : 'Decreased predicted risk'}</span>
      </div>
      <code>{factor.shap_impact > 0 ? '+' : ''}{factor.shap_impact.toFixed(3)}</code>
    </li>
  );
}

function ShapImpactPlot({ factors }) {
  const maxImpact = Math.max(...factors.map((factor) => Math.abs(factor.shap_impact)), 0.001);

  return (
    <section className="shap-block">
      <div className="section-title-row">
        <h3>SHAP Impact Plot</h3>
        <BarChart3 size={18} aria-hidden="true" />
      </div>
      <div className="shap-axis" aria-hidden="true">
        <span>Decreases risk</span>
        <i />
        <span>Increases risk</span>
      </div>
      <ol className="shap-list">
        {factors.map((factor) => {
          const value = factor.shap_impact;
          const increased = value > 0;
          const width = `${Math.max(8, (Math.abs(value) / maxImpact) * 50)}%`;

          return (
            <li className="shap-row" key={`plot-${factor.feature}-${value}`}>
              <span>{factor.display_feature}</span>
              <div className="shap-track">
                <i className={increased ? 'positive' : 'negative'} style={{ width }} />
              </div>
              <code>{value > 0 ? '+' : ''}{value.toFixed(3)}</code>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function App() {
  const [step, setStep] = useState('landing');
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState('');

  const probability = result?.probability ?? 0;
  const meterStyle = useMemo(() => ({ '--risk': `${Math.max(2, Math.round(probability * 100))}%` }), [probability]);

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function submit(event) {
    event.preventDefault();
    setStatus('loading');
    setError('');

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(toPayload(form)),
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || `Request failed with ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setStatus('success');
    } catch (requestError) {
      setStatus('error');
      setError(requestError.message || 'Could not reach the prediction API.');
    }
  }

  function resetForm() {
    setForm(initialForm);
    setResult(null);
    setError('');
    setStatus('idle');
  }

  if (step === 'landing') {
    return <LandingPage onStart={() => setStep('assessment')} />;
  }

  return (
    <main className="app-shell assessment-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Clinical Decision Support</p>
            <h1>CardioRisk</h1>
          </div>
          <div className="topbar-actions">
            <button className="ghost-button" type="button" onClick={() => setStep('landing')}>
              Overview
            </button>
            <div className="api-status">
              <CheckCircle2 size={18} aria-hidden="true" />
              <span>Prediction Engine Ready</span>
            </div>
          </div>
        </header>

        <div className="layout-grid">
          <form className="form-panel" onSubmit={submit}>
            <div className="panel-heading">
              <div>
                <h2>Prediction Inputs</h2>
                <p>Complete the patient profile to generate an explained model output.</p>
              </div>
              <HeartPulse size={24} aria-hidden="true" />
            </div>

            <div className="example-strip" aria-label="Example patients">
              {examples.map((example) => (
                <button type="button" key={example.name} onClick={() => setForm(example.values)}>
                  {example.name}
                </button>
              ))}
            </div>

            {fieldGroups.map((group) => (
              <section className="field-section" key={group.title}>
                <h3>{group.title}</h3>
                <div className="field-grid">
                  {group.fields.map((field) => (
                    <Field key={field.key} field={field} value={form[field.key]} onChange={updateField} />
                  ))}
                </div>
              </section>
            ))}

            {error && (
              <div className="error-box" role="alert">
                <AlertTriangle size={18} aria-hidden="true" />
                <span>{error}</span>
              </div>
            )}

            <div className="actions">
              <button className="secondary" type="button" onClick={resetForm}>
                <RotateCcw size={18} aria-hidden="true" />
                Reset
              </button>
              <button className="primary" type="submit" disabled={status === 'loading'}>
                {status === 'loading' ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <Send size={18} aria-hidden="true" />}
                {status === 'loading' ? 'Predicting' : 'Run Prediction'}
              </button>
            </div>
          </form>

          <aside className="results-panel" aria-live="polite">
            <div className="result-header">
              <div>
                <p className="eyebrow">Model Output</p>
                <h2>{result ? riskLabel(probability) : 'Awaiting input'}</h2>
              </div>
              <Activity size={25} aria-hidden="true" />
            </div>

            <div className="risk-meter" style={meterStyle}>
              <div className="risk-number">{result ? formatPercent(probability) : '--'}</div>
              <div className="meter-track"><span /></div>
              <div className="meter-labels"><span>0%</span><span>50%</span><span>100%</span></div>
            </div>

            {result ? (
              <>
                <section className="summary-block">
                  <h3>Summary</h3>
                  <p>{result.summary}</p>
                </section>

                <ShapImpactPlot factors={result.top_factors} />

                <section className="factor-block">
                  <h3>Top Factors</h3>
                  <ol>
                    {result.top_factors.map((factor) => (
                      <Factor key={`${factor.feature}-${factor.shap_impact}`} factor={factor} />
                    ))}
                  </ol>
                </section>

                <section className={result.warnings.length ? 'warning-block active' : 'warning-block'}>
                  <h3>Warnings</h3>
                  {result.warnings.length ? (
                    result.warnings.map((warning) => <p key={warning}>{warning}</p>)
                  ) : (
                    <p>No input reliability warnings.</p>
                  )}
                </section>
              </>
            ) : (
              <div className="empty-state">
                <HeartPulse size={36} aria-hidden="true" />
                <p>Run a prediction to see probability, SHAP factors, warnings, and the model summary.</p>
              </div>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}

export default App;
