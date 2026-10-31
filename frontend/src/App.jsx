import React, { useState, useEffect, useRef } from 'react';
import { Shield, Radio, Terminal, Cpu, HardDrive, RefreshCw } from 'lucide-react';
import CallScreen from './components/CallScreen';
import ScammerMap from './components/ScammerMap';
import MockBankingApp from './components/MockBankingApp';

export default function App() {
  const [telemetry, setTelemetry] = useState({
    session_id: 'init_sess',
    transcript: 'Standby: VaniShield AI active. Ready for live call packets...',
    acoustic_clone: 0.1,
    intent_scam: 0.0,
    unified_risk: 0.1,
    threat_level: 'SECURE',
    circuit_breaker_locked: false,
    caller_geo: {
      number: '+91 140 998 8776',
      carrier: 'Telemarketer / VoIP Gateway (Prefix 140)',
      city: 'New Delhi Circle',
      lat: 28.6139,
      lon: 77.2090,
      is_scam: false,
    },
    waveform: [10, 20, 30, 20, 10, 15, 25, 35, 25, 15, 10, 20, 30, 25, 15, 10]
  });

  const [bankingState, setBankingState] = useState({
    locked: false,
    reason: 'Account Normal',
    risk_score: 0.0,
  });

  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  // Initialize WebSocket connection to backend
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || '127.0.0.1';
    const wsUrl = `${protocol}//${host}:8000/ws/live-call`;

    // Synchronize initial banking circuit-breaker status
    const port = window.location.port === '5173' || window.location.port === '3000' ? '8000' : (window.location.port || '8000');
    const httpProtocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    fetch(`${httpProtocol}//${host}:${port}/api/banking/status`)
      .then(res => res.json())
      .then(data => {
        if (data.locked) {
          setBankingState({
            locked: true,
            reason: data.reason || 'Account under security hold.',
            risk_score: data.risk_score || 85.0
          });
          setTelemetry(prev => ({ ...prev, circuit_breaker_locked: true }));
        }
      })
      .catch(() => {});

    let ws;
    try {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'SESSION_INIT') {
            if (msg.circuit_breaker) {
              setBankingState(prev => ({
                ...prev,
                locked: true,
                reason: 'Account Protection: Previous high-risk scam triggered circuit breaker.'
              }));
              setTelemetry(prev => ({ ...prev, circuit_breaker_locked: true }));
            }
          } else if (msg.type === 'TELEMETRY_FRAME') {
            setTelemetry(prev => ({
              ...prev,
              ...msg,
              caller_geo: msg.caller_geo || prev.caller_geo
            }));
            if (msg.circuit_breaker_locked) {
              setBankingState({
                locked: true,
                reason: `Fraud Defense: Voice Clone Risk (${msg.unified_risk}%) detected on live call.`,
                risk_score: msg.unified_risk
              });
            }
          }
        } catch (e) {
          console.error('Failed to parse frame:', e);
        }
      };

      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
    } catch (e) {
      setWsConnected(false);
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const handleSimulateScam = () => {
    if (wsRef.current && wsConnected) {
      wsRef.current.send(JSON.stringify({ action: 'SIMULATE_SCAM_CALL' }));
    } else {
      // Fallback local animation if offline
      setTelemetry(prev => ({
        ...prev,
        acoustic_clone: 99.4,
        intent_scam: 96.0,
        unified_risk: 97.9,
        threat_level: 'CRITICAL SCAM (LOCKED)',
        circuit_breaker_locked: true,
        transcript: 'Urgent security notice from State Bank of India fraud team. Share your 6-digit OTP and CVV immediately to block illegal debit.',
        caller_geo: {
          number: '+91 140 998 8776',
          carrier: 'Virtual VoIP Gateway (Flagged)',
          city: 'Jamtara / Cyber Node',
          lat: 23.9632,
          lon: 86.8029,
          is_scam: true
        }
      }));
      setBankingState({
        locked: true,
        reason: 'Fraud Defense: Unified Risk 97.9% — AI Voice Clone Scam Intercepted.',
        risk_score: 97.9
      });
    }
  };

  const handleSimulateSafe = () => {
    if (wsRef.current && wsConnected) {
      wsRef.current.send(JSON.stringify({ action: 'SIMULATE_SAFE_CALL' }));
    } else {
      setTelemetry(prev => ({
        ...prev,
        acoustic_clone: 0.2,
        intent_scam: 0.0,
        unified_risk: 0.1,
        threat_level: 'AUTHENTIC (SAFE)',
        circuit_breaker_locked: false,
        transcript: 'Hi Rahul, are we still meeting for lunch today at 1 PM? Let me know when you reach.',
        caller_geo: {
          number: '+91 98765 43210',
          carrier: 'Reliance Jio Infocomm',
          city: 'Bengaluru Circle',
          lat: 12.9716,
          lon: 77.5946,
          is_scam: false
        }
      }));
    }
  };

  const handleUnlock = async (pin) => {
    try {
      const host = window.location.hostname || '127.0.0.1';
      const port = window.location.port === '5173' || window.location.port === '3000' ? '8000' : (window.location.port || '8000');
      const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
      const res = await fetch(`${protocol}//${host}:${port}/api/banking/unlock`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pin: pin || '1234' })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setBankingState({
          locked: false,
          reason: 'Biometrically authorized user release',
          risk_score: 0.0
        });
        setTelemetry(prev => ({ ...prev, circuit_breaker_locked: false }));
        return true;
      } else {
        alert(data.detail || data.message || 'Invalid Security PIN.');
        return false;
      }
    } catch (err) {
      if (pin === '1234') {
        setBankingState({
          locked: false,
          reason: 'Biometrically authorized user release (local fallback)',
          risk_score: 0.0
        });
        setTelemetry(prev => ({ ...prev, circuit_breaker_locked: false }));
        return true;
      } else {
        alert('Invalid Security PIN. Default test PIN is 1234.');
        return false;
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-gray-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <header className="bg-[#111827] border-b border-[#1f2937] px-6 py-3 flex items-center justify-between sticky top-0 z-30 shadow-md">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-600/20 border border-blue-500/30 rounded-lg text-blue-400">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold tracking-tight text-white">Project VaniShield AI</h1>
              <span className="px-2 py-0.5 text-[10px] font-semibold bg-blue-500/20 text-blue-300 rounded border border-blue-500/30">
                Team CypherEye
              </span>
            </div>
            <p className="text-[11px] text-gray-400">Real-Time Voice Cloning Defense & Automated Banking Circuit Breaker</p>
          </div>
        </div>

        {/* Telemetry Hardware Badges */}
        <div className="flex items-center space-x-3 text-xs">
          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0b0f19] border border-gray-800 rounded-lg">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-gray-400">GPU:</span>
            <span className="text-white font-mono font-medium">RTX 3050 (FP16)</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0b0f19] border border-gray-800 rounded-lg">
            <Radio className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-gray-400">Jev-Omni 12B:</span>
            <span className="text-cyan-300 font-mono font-medium">Port 9931</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0b0f19] border border-gray-800 rounded-lg">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span className="text-gray-400">Stream:</span>
            <span className={`font-mono font-medium ${wsConnected ? 'text-emerald-300' : 'text-amber-300'}`}>
              {wsConnected ? 'ONLINE' : 'STANDBY'}
            </span>
          </div>
        </div>
      </header>

      {/* Main 3-Panel Split-Screen Workspace */}
      <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-3 gap-6 max-w-[1700px] w-full mx-auto">
        {/* Panel 1: Live Call Forensic Card & Audio Meter */}
        <div className="h-full">
          <CallScreen
            telemetry={telemetry}
            onSimulateScam={handleSimulateScam}
            onSimulateSafe={handleSimulateSafe}
            isStreaming={wsConnected}
          />
        </div>

        {/* Panel 2: Interactive Leaflet Threat Geolocation Map */}
        <div className="h-full">
          <ScammerMap
            callerGeo={telemetry.caller_geo}
          />
        </div>

        {/* Panel 3: Protected Mock Banking Portal & UPI Circuit Breaker */}
        <div className="h-full">
          <MockBankingApp
            bankingState={bankingState}
            onUnlock={handleUnlock}
          />
        </div>
      </main>

      {/* Bottom Footer */}
      <footer className="bg-[#111827] border-t border-[#1f2937] px-6 py-2.5 text-xs text-gray-500 flex justify-between items-center">
        <span>VaniShield AI Architecture v2.0 | Dual-Brain Forensic Arbitration (Wav2Vec 2.0 + Jev-Omni 12B)</span>
        <span>India DPDP Act 2023 Compliant In-Memory Volatile Buffer (&lt;4.0s SLA)</span>
      </footer>
    </div>
  );
}
