import React, { useState } from 'react';
import { Landmark, Lock, Unlock, ArrowRight, ShieldCheck, AlertOctagon, CheckCircle2 } from 'lucide-react';

export default function MockBankingApp({ bankingState, onUnlock, onTransfer }) {
  const [amount, setAmount] = useState('45000');
  const [payee, setPayee] = useState('Electricity Board / KYC Agent');
  const [transferMsg, setTransferMsg] = useState(null);
  const [showPinModal, setShowPinModal] = useState(false);
  const [pinInput, setPinInput] = useState('');

  const isLocked = bankingState?.locked;

  const handlePay = () => {
    if (isLocked) return;
    const res = onTransfer ? onTransfer(payee, amount) : { success: true };
    setTransferMsg(`Rs. ${parseFloat(amount).toLocaleString('en-IN')} successfully transferred.`);
    setTimeout(() => setTransferMsg(null), 4000);
  };

  const handleUnlockSubmit = async (e) => {
    e.preventDefault();
    if (onUnlock) {
      const ok = await onUnlock(pinInput);
      if (ok !== false) {
        setShowPinModal(false);
        setPinInput('');
      }
    }
  };

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-5 shadow-2xl flex flex-col justify-between h-full relative">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-gray-800">
          <div className="flex items-center space-x-3">
            <div className={`p-2.5 rounded-lg ${isLocked ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
              <Landmark className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Protected Bank & UPI Portal</h2>
              <p className="text-xs text-gray-400">Automated Banking Circuit Breaker</p>
            </div>
          </div>
          <span className={`px-2.5 py-1 text-xs font-semibold rounded-full flex items-center gap-1.5 ${
            isLocked ? 'bg-red-500/20 text-red-400 border border-red-500/40' : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
          }`}>
            {isLocked ? <Lock className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
            {isLocked ? 'PAYMENTS FROZEN' : 'ACTIVE & SECURE'}
          </span>
        </div>

        {/* Account Balance Card */}
        <div className="mt-4 p-4 bg-gradient-to-br from-gray-900 to-[#0b0f19] border border-gray-800 rounded-xl relative overflow-hidden">
          <div className="flex justify-between items-start">
            <div>
              <div className="text-xs text-gray-400">Primary Savings Account</div>
              <div className="text-xl font-bold font-mono text-white mt-0.5">Rs. 2,84,500.00</div>
              <div className="text-[11px] text-gray-500 font-mono mt-1">HDFC-****-****-4921</div>
            </div>
            <span className="text-[10px] font-semibold tracking-wider text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">
              UPI 2.0
            </span>
          </div>
        </div>

        {/* Circuit Breaker Status Banner */}
        {isLocked && (
          <div className="mt-4 p-3 bg-red-500/15 border border-red-500/40 rounded-lg animate-pulse">
            <div className="flex items-center gap-2 text-red-400 text-xs font-bold">
              <AlertOctagon className="w-4 h-4 flex-shrink-0" />
              <span>CIRCUIT BREAKER ENGAGED: UPI & NET BANKING LOCKED</span>
            </div>
            <p className="text-[11px] text-gray-300 mt-1 leading-snug">
              {bankingState.reason || 'High-Risk AI Voice Clone Fraud detected during active phone call. Payment gateways biometrically frozen.'}
            </p>
          </div>
        )}

        {/* Mock UPI Payment Form */}
        <div className="mt-4 space-y-3">
          <div>
            <label className="text-[11px] text-gray-400 block mb-1">Payee UPI ID / Contact:</label>
            <input
              type="text"
              disabled={isLocked}
              value={payee}
              onChange={(e) => setPayee(e.target.value)}
              className="w-full bg-[#0b0f19] border border-gray-800 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </div>

          <div>
            <label className="text-[11px] text-gray-400 block mb-1">Transfer Amount (INR):</label>
            <input
              type="text"
              disabled={isLocked}
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-full bg-[#0b0f19] border border-gray-800 rounded-lg px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </div>

          {transferMsg && (
            <div className="p-2 bg-emerald-500/15 border border-emerald-500/40 rounded-lg text-emerald-400 text-xs flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
              <span>{transferMsg}</span>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mt-5 pt-4 border-t border-gray-800 space-y-2">
        <button
          onClick={handlePay}
          disabled={isLocked}
          className={`w-full py-2.5 px-4 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all ${
            isLocked
              ? 'bg-red-500/20 text-red-400 border border-red-500/30 cursor-not-allowed opacity-90'
              : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white shadow-lg'
          }`}
        >
          {isLocked ? (
            <>
              <Lock className="w-4 h-4" /> PAYMENTS LOCKED (RISK &ge; 70%)
            </>
          ) : (
            <>
              Pay via UPI <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>

        {isLocked && (
          <button
            onClick={() => setShowPinModal(true)}
            className="w-full py-2 px-3 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors border border-gray-700"
          >
            <Unlock className="w-3.5 h-3.5 text-amber-400" /> Biometric / PIN Emergency Release
          </button>
        )}
      </div>

      {/* Emergency PIN Modal */}
      {showPinModal && (
        <div className="absolute inset-0 bg-black/80 backdrop-blur-sm rounded-xl p-6 flex flex-col justify-center items-center z-20">
          <div className="bg-[#111827] border border-gray-700 rounded-xl p-5 w-full max-w-xs shadow-2xl">
            <h3 className="text-sm font-bold text-white mb-1 flex items-center gap-1.5">
              <Unlock className="w-4 h-4 text-amber-400" /> Authorization Release
            </h3>
            <p className="text-[11px] text-gray-400 mb-3">Enter user security PIN (Default: 1234) to unlock UPI services.</p>
            <form onSubmit={handleUnlockSubmit} className="space-y-3">
              <input
                type="password"
                maxLength="4"
                autoFocus
                placeholder="****"
                value={pinInput}
                onChange={(e) => setPinInput(e.target.value)}
                className="w-full bg-[#0b0f19] border border-gray-700 rounded-lg px-3 py-2 text-center text-lg font-mono tracking-widest text-white focus:outline-none focus:border-amber-400"
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setShowPinModal(false)}
                  className="w-1/2 py-2 bg-gray-800 text-gray-300 rounded-lg text-xs hover:bg-gray-700"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="w-1/2 py-2 bg-amber-600 text-white rounded-lg text-xs font-bold hover:bg-amber-500"
                >
                  Unlock
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
