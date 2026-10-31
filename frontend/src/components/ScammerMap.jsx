import React, { useEffect, useRef } from 'react';
import { MapPin, Globe, Shield } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export default function ScammerMap({ callerGeo, blacklist }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);

  useEffect(() => {
    // Dynamically initialize Leaflet map
    if (typeof window !== 'undefined' && mapContainerRef.current) {
      if (!mapInstanceRef.current) {
        const lat = callerGeo?.lat || 28.6139;
        const lon = callerGeo?.lon || 77.2090;

        const map = L.map(mapContainerRef.current, {
          center: [lat, lon],
          zoom: 5,
          zoomControl: false,
          attributionControl: false
        });

        // Dark theme map tiles
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
          maxZoom: 19,
          subdomains: 'abcd',
        }).addTo(map);

        mapInstanceRef.current = map;
      }
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (mapInstanceRef.current && callerGeo) {
      const lat = callerGeo.lat || 28.6139;
      const lon = callerGeo.lon || 77.2090;
      const isScam = callerGeo.is_scam;

      mapInstanceRef.current.setView([lat, lon], isScam ? 7 : 5, { animate: true });

      if (markerRef.current) {
        markerRef.current.remove();
      }

      const iconHtml = `<div class="w-6 h-6 rounded-full flex items-center justify-center ${
        isScam ? 'bg-red-500 ring-4 ring-red-500/30' : 'bg-emerald-500 ring-4 ring-emerald-500/30'
      } text-white shadow-lg animate-bounce">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>
      </div>`;

      const customIcon = L.divIcon({
        className: 'custom-pin',
        html: iconHtml,
        iconSize: [24, 24],
        iconAnchor: [12, 24]
      });

      markerRef.current = L.marker([lat, lon], { icon: customIcon })
        .addTo(mapInstanceRef.current)
        .bindPopup(`
          <div style="font-family: sans-serif; font-size: 12px; color: #111;">
            <strong>${isScam ? '⚠️ Flagged Scammer Node' : '✅ Verified Caller'}</strong><br/>
            Number: ${callerGeo.number || 'Unknown'}<br/>
            Circle: ${callerGeo.city || 'India'}<br/>
            Carrier: ${callerGeo.carrier || 'VoIP Gateway'}
          </div>
        `)
        .openPopup();
    }
  }, [callerGeo]);

  return (
    <div className="bg-[#111827] border border-[#1f2937] rounded-xl p-5 shadow-2xl flex flex-col justify-between h-full">
      <div>
        <div className="flex items-center justify-between pb-4 border-b border-gray-800">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-lg bg-indigo-500/20 text-indigo-400">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Live Telecom Threat Map</h2>
              <p className="text-xs text-gray-400">Origin Node & Carrier Network Triangulation</p>
            </div>
          </div>
          <span className="text-xs text-indigo-400 font-mono flex items-center gap-1">
            <MapPin className="w-3.5 h-3.5" /> India National Grid
          </span>
        </div>

        {/* Map Container */}
        <div className="mt-4 relative rounded-lg overflow-hidden border border-gray-800 h-64 bg-[#0b0f19]">
          <div ref={mapContainerRef} className="w-full h-full" />
        </div>

        {/* Telecom Metadata Strip */}
        <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
          <div className="bg-[#0b0f19] p-2.5 rounded-lg border border-gray-800">
            <div className="text-gray-500 text-[10px]">Active Node</div>
            <div className="font-semibold text-gray-200 truncate">{callerGeo?.city || 'New Delhi'}</div>
          </div>
          <div className="bg-[#0b0f19] p-2.5 rounded-lg border border-gray-800">
            <div className="text-gray-500 text-[10px]">ISP / Telecom</div>
            <div className="font-semibold text-gray-200 truncate">{callerGeo?.carrier || 'VoIP SIP'}</div>
          </div>
          <div className="bg-[#0b0f19] p-2.5 rounded-lg border border-gray-800">
            <div className="text-gray-500 text-[10px]">Grid Status</div>
            <div className={`font-semibold ${callerGeo?.is_scam ? 'text-red-400' : 'text-emerald-400'}`}>
              {callerGeo?.is_scam ? 'THREAT PINNED' : 'INTERCEPT SAFE'}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-gray-800 text-[11px] text-gray-500 flex items-center justify-between">
        <span className="flex items-center gap-1"><Shield className="w-3.5 h-3.5 text-blue-400" /> Carrier SS7 / HLR Gateway</span>
        <span>DPDP Act 2023 Compliant</span>
      </div>
    </div>
  );
}
