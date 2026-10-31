import React from 'react';
import { Phone, PhoneCall, ShieldAlert, ShieldCheck, Activity, Radio, AlertTriangle } from 'lucide-react';

export default function CallScreen({ telemetry, onSimulateScam, onSimulateSafe, isStreaming }) {
  const risk = telemetry.unified_risk || 0;
  const isScam = risk >= 70;
  const acoustic = telemetry.acoustic_clone || 0;
  const intent = telemetry.intent_scam || 0;

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-5 shadow-2xl flex flex-col justify-between h-full">
      {/* Header */}
      <div>
        <div className="flex items-center justify-between pb-4 border-b border-gray-800">
          <div className="flex items-center space-x-3">
            <div className={`p-2.5 rounded-lg ${isScam ? 'bg-red-500/20 text-red-400' : 'bg-blue-500/20 text-blue-400'}`}>
              <PhoneCall className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Live Call Forensic Monitor</h2>
              <p className="text-xs text-gray-400">Low-Latency Acoustic & Intent Interceptor</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!isStreaming && (
              <span className="px-2 py-0.5 text-[10px] font-medium text-gray-400 bg-gray-800/60 rounded border border-gray-700/50">
                DISCONNECTED
              </span>
            )}
            <span className={`px-2.5 py-1 text-xs font-semibold rounded-full flex items-center gap-1.5 ${
              isScam
                ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                : !isStreaming
                ? 'bg-gray-500/20 text-gray-400 border border-gray-500/40'
                : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
            }`}>
              <span className={`w-2 h-2 rounded-full ${
                isScam ? 'bg-red-400 animate-ping' : !isStreaming ? 'bg-gray-400' : 'bg-emerald-400'
              }`}></span>
              {isScam ? 'SCAM DETECTED' : !isStreaming ? 'STANDBY / IDLE' : 'SECURE CALL'}
            </span>
          </div>
        </div>

        {/* Caller Telecom Profile */}
        <div className="mt-4 p-3 bg-[#0b0f19] border border-gray-800 rounded-lg">
          <div className="flex justify-between items-center text-xs">
            <span className="text-gray-400">Caller Identity:</span>
            <span className="text-white font-mono font-medium">{telemetry.caller_geo?.number || '+91 140 998 8776'}</span>
          </div>
          <div className="flex justify-between items-center text-xs mt-1.5">
            <span className="text-gray-400">Telecom Carrier:</span>
            <span className="text-blue-400 font-medium">{telemetry.caller_geo?.carrier || 'VoIP Gateway / Telemarketer'}</span>
          </div>
          <div className="flex justify-between items-center text-xs mt-1.5">
            <span className="text-gray-400">Origin Circle:</span>
            <span className="text-gray-300 font-medium">{telemetry.caller_geo?.city || 'New Delhi Circle'}</span>
          </div>
        </div>

        {/* Dynamic Waveform Visualizer */}
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
            <span className="flex items-center gap-1"><Activity className="w-3.5 h-3.5 text-blue-400" /> Audio Stream Waveform</span>
            <span className="font-mono text-gray-500">16,000 Hz PCM</span>
          </div>
          <div className="h-16 bg-[#0b0f19] border border-gray-800 rounded-lg flex items-center justify-center gap-1 px-3 overflow-hidden">
            {(telemetry.waveform || [15, 30, 45, 60, 35, 20, 80, 50, 40, 25, 65, 30, 45, 75, 55, 35]).map((val, idx) => {
              const h = Math.max(8, Math.min(52, Math.abs(val) * (isScam ? 50 : 25)));
              return (
                <div
                  key={idx}
                  className={`w-1.5 rounded-full transition-all duration-150 ${
                    isScam ? 'bg-red-500' : 'bg-blue-500'
                  }`}
                  style={{ height: `${h}px` }}
                />
              );
            })}
          </div>
        </div>

        {/* Dual-Brain Risk Gauges */}
        <div className="mt-5 space-y-3">
          {/* Unified Risk Score */}
          <div>
            <div className="flex justify-between items-center text-xs mb-1">
              <span className="text-gray-300 font-semibold">Unified Scam Risk Score</span>
              <span className={`font-mono font-bold text-sm ${isScam ? 'text-red-400' : 'text-emerald-400'}`}>
                {risk.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
              <div
                className={`h-2.5 rounded-full transition-all duration-300 ${
                  isScam ? 'bg-gradient-to-r from-orange-500 to-red-600' : 'bg-gradient-to-r from-blue-500 to-emerald-500'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, risk))}%` }}
              />
            </div>
          </div>

          {/* Sub-scores (Wav2Vec2 + Jev-Omni) */}
          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="bg-[#0b0f19] p-2.5 rounded-lg border border-gray-800">
              <div className="text-[11px] text-gray-400">Acoustic Wav2Vec2</div>
              <div className="text-base font-bold font-mono text-cyan-400">{acoustic.toFixed(1)}%</div>
              <div className="text-[10px] text-gray-500">Physical Glottal DSP</div>
            </div>
            <div className="bg-[#0b0f19] p-2.5 rounded-lg border border-gray-800">
              <div className="text-[11px] text-gray-400">Jev-Omni 12B Intent</div>
              <div className="text-base font-bold font-mono text-amber-400">{intent.toFixed(1)}%</div>
              <div className="text-[10px] text-gray-500">Semantic Fraud Vishing</div>
            </div>
          </div>
        </div>

        {/* Live ASR Forensic Transcript */}
        <div className="mt-4">
          <div className="text-xs text-gray-400 mb-1.5 flex items-center gap-1">
            <Radio className="w-3.5 h-3.5 text-emerald-400" />
            Live Call Audio Transcript:
          </div>
          <div className="p-3 bg-[#0b0f19] border border-gray-800 rounded-lg text-xs font-mono text-gray-200 min-h-[56px] leading-relaxed">
            {telemetry.transcript || 'Listening for live voice packets on WebSocket...'}
          </div>
        </div>
      </div>

      {/* Action Simulation Controls */}
      <div className="mt-5 pt-4 border-t border-gray-800 grid grid-cols-2 gap-2">
        <button
          onClick={onSimulateScam}
          className="px-3 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-500/40 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
        >
          <ShieldAlert className="w-4 h-4" /> Simulate Scam Attack
        </button>
        <button
          onClick={onSimulateSafe}
          className="px-3 py-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 border border-blue-500/40 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
        >
          <ShieldCheck className="w-4 h-4" /> Simulate Safe Call
        </button>
      </div>
    </div>
  );
}
