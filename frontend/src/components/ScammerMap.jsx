import React, { useEffect, useRef } from 'react';
import { MapPin, Globe, Shield } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

export default function ScammerMap({ callerGeo, blacklist }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markerRef = useRef(null);
  const circleRef = useRef(null);

  useEffect(() => {
    // Dynamically initialize Leaflet map
    if (typeof window !== 'undefined' && mapContainerRef.current) {
      if (!mapInstanceRef.current) {
        const lat = Number(callerGeo?.lat) || 28.6139;
        const lon = Number(callerGeo?.lon) || 77.2090;

        const map = L.map(mapContainerRef.current, {
          center: [lat, lon],
          zoom: 5,
          zoomControl: false,
          attributionControl: false
        });

        // Dark theme map tiles (Supports Mapbox token or free CartoDB fallback)
        const mapboxToken = import.meta.env?.VITE_MAPBOX_ACCESS_TOKEN || import.meta.env?.MAPBOX_ACCESS_TOKEN;
        const tileUrl = mapboxToken
          ? `https://api.mapbox.com/styles/v1/mapbox/dark-v11/tiles/{z}/{x}/{y}?access_token=${mapboxToken}`
          : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';

        L.tileLayer(tileUrl, {
          maxZoom: 19,
          subdomains: mapboxToken ? undefined : 'abcd',
          attribution: mapboxToken ? '© Mapbox' : '© CartoDB',
        }).addTo(map);

        mapInstanceRef.current = map;

        setTimeout(() => {
          map.invalidateSize();
        }, 250);
      }
    }

    const handleResize = () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (mapInstanceRef.current && callerGeo) {
      const lat = Number(callerGeo.lat) || 28.6139;
      const lon = Number(callerGeo.lon) || 77.2090;
      const isScam = Boolean(callerGeo.is_scam);

      mapInstanceRef.current.invalidateSize();
      mapInstanceRef.current.setView([lat, lon], isScam ? 7 : 6, { animate: true });

      if (markerRef.current) {
        markerRef.current.remove();
      }
      if (circleRef.current) {
        circleRef.current.remove();
      }

      // Add threat radar perimeter
      circleRef.current = L.circle([lat, lon], {
        radius: isScam ? 95000 : 35000,
        color: isScam ? '#ef4444' : '#10b981',
        weight: 1.5,
        fillColor: isScam ? '#ef4444' : '#10b981',
        fillOpacity: isScam ? 0.22 : 0.12,
        dashArray: isScam ? '4, 4' : null
      }).addTo(mapInstanceRef.current);

      const iconHtml = `<div class="w-7 h-7 rounded-full flex items-center justify-center ${
        isScam ? 'bg-red-500 ring-4 ring-red-500/40 shadow-red-500/50' : 'bg-emerald-500 ring-4 ring-emerald-500/40 shadow-emerald-500/50'
      } text-white shadow-xl animate-pulse">
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>
      </div>`;

      const customIcon = L.divIcon({
        className: 'custom-pin',
        html: iconHtml,
        iconSize: [28, 28],
        iconAnchor: [14, 14]
      });

      markerRef.current = L.marker([lat, lon], { icon: customIcon })
        .addTo(mapInstanceRef.current)
        .bindPopup(`
          <div style="font-family: ui-monospace, monospace; font-size: 11px; color: #111; min-width: 175px;">
            <strong style="color: ${isScam ? '#dc2626' : '#059669'}; font-size: 12px; display: block; margin-bottom: 3px;">
              ${isScam ? '⚠️ FLAGGED SCAMMER NODE' : '✅ VERIFIED CALLER CIRCLE'}
            </strong>
            <div><strong>Number:</strong> ${callerGeo.number || 'Unknown'}</div>
            <div><strong>Circle:</strong> ${callerGeo.city || 'India Telecom'}</div>
            <div><strong>Carrier:</strong> ${callerGeo.carrier || 'VoIP Gateway'}</div>
            <div><strong>Coordinates:</strong> ${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E</div>
            <div style="margin-top: 4px; padding: 2px 4px; border-radius: 4px; font-weight: bold; font-size: 10px; background: ${isScam ? '#fee2e2' : '#d1fae5'}; color: ${isScam ? '#991b1b' : '#065f46'};">
              ${isScam ? 'INTERLOCK: CIRCUIT BREAKER ARMED' : 'STATUS: NORMAL OPERATION'}
            </div>
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
