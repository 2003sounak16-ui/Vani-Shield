"""
VaniShield AI — Caller Tracker, Carrier Lookup & Geolocation Module
Team CypherEye | Project VaniShield AI
Complies with India's DPDP Act 2023 with atomic persistence and caller pseudonymization.
"""

import os
import sys
import json
import time
import hmac
import hashlib
import threading
import argparse
import requests

try:
    import phonenumbers
    from phonenumbers import carrier, geocoder, timezone
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_curr_dir) == "engine":
    PROJECT_ROOT = os.path.abspath(os.path.join(_curr_dir, "..", ".."))
else:
    PROJECT_ROOT = _curr_dir
BLACKLIST_PATH = os.environ.get("VANISHIELD_BLACKLIST", os.path.join(PROJECT_ROOT, "scammer_blacklist.json"))
_blacklist_lock = threading.Lock()

# Regional fallbacks for Indian telecommunication circles & metro nodes
REGIONAL_NODES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi", "country": "India"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "country": "India"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "country": "India"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "country": "India"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "country": "India"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "country": "India"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "country": "India"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "country": "India"},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "country": "India"},
    "Jamtara": {"lat": 23.9632, "lon": 86.8029, "state": "Jharkhand", "country": "India"},
}

def get_caller_hash_secret() -> bytes:
    """Resolve salt/secret with priority for VANISHIELD_HMAC_SECRET then VANISHIELD_SALT."""
    secret = os.environ.get("VANISHIELD_HMAC_SECRET", os.environ.get("VANISHIELD_SALT", "vanishield-dpdp-2023-secure-salt"))
    return secret.encode("utf-8") if isinstance(secret, str) else secret

def hash_caller_identity(phone_number: str) -> str:
    """DPDP Act 2023 compliant HMAC-SHA256 caller identity pseudonymizer (canonical digits)."""
    if not phone_number:
        return "ANONYMIZED_CALLER"
    clean = "".join(filter(str.isdigit, str(phone_number).strip()))
    secret = get_caller_hash_secret()
    return hmac.new(secret, clean.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

def hash_caller_identity_legacy(phone_number: str) -> str:
    """Legacy hash without digits stripping for backward compatibility with older saved records."""
    if not phone_number:
        return "ANONYMIZED_CALLER"
    secret = get_caller_hash_secret()
    return hmac.new(secret, str(phone_number).strip().encode("utf-8"), hashlib.sha256).hexdigest()[:16]

def mask_phone(phone_number: str) -> str:
    """Safely mask caller phone number for DPDP-compliant privacy preservation."""
    clean = phone_number.strip()
    if len(clean) >= 8:
        return f"{clean[:4]}****{clean[-4:]}"
    return "ANONYMIZED_CALLER" 

def lookup_carrier(phone_number: str) -> dict:
    """Extract carrier, geo-region, and valid telecom status from a phone number."""
    result = {
        "number": phone_number,
        "valid": False,
        "carrier": "Unknown Carrier / Virtual SIP",
        "location": "National / Unknown",
        "timezones": [],
        "lat": 28.6139,
        "lon": 77.2090,
    }

    if not PHONENUMBERS_AVAILABLE:
        return result

    try:
        # Default region India (+91) if not explicitly provided
        default_region = "IN" if not phone_number.startswith("+") else None
        parsed = phonenumbers.parse(phone_number, default_region)
        result["valid"] = phonenumbers.is_valid_number(parsed)

        carrier_name = carrier.name_for_number(parsed, "en")
        if carrier_name:
            result["carrier"] = carrier_name
        elif phone_number.startswith("+91140") or phone_number.startswith("140"):
            result["carrier"] = "Telemarketer / VoIP Gateway (Prefix 140)"

        geo_desc = geocoder.description_for_number(parsed, "en")
        if geo_desc:
            result["location"] = geo_desc
            for city_key, node in REGIONAL_NODES.items():
                if city_key.lower() in geo_desc.lower():
                    result["lat"] = node["lat"]
                    result["lon"] = node["lon"]
                    break

        tzs = timezone.time_zones_for_number(parsed)
        result["timezones"] = list(tzs) if tzs else ["Asia/Kolkata"]

    except Exception as e:
        result["error"] = str(e)

    return result

def get_ip_geolocation(ip_address: str = None) -> dict:
    """Fetch geo-coordinates via IP lookup, with safe offline fallback."""
    if not ip_address or ip_address in ("127.0.0.1", "localhost", "::1"):
        # Local node fallback: New Delhi Cyber Defense Hub
        return {"city": "New Delhi", "region": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090, "isp": "Local Test Interface"}

    try:
        resp = requests.get(f"http://ip-api.com/json/{ip_address}?fields=status,country,regionName,city,lat,lon,isp", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return {
                    "city": data.get("city", "Unknown City"),
                    "region": data.get("regionName", "Unknown Region"),
                    "country": data.get("country", "India"),
                    "lat": float(data.get("lat", 28.6139)),
                    "lon": float(data.get("lon", 77.2090)),
                    "isp": data.get("isp", "Telecom Provider")
                }
    except Exception:
        pass

    return {"city": "New Delhi", "region": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090, "isp": "Telecom Gateway"}

def load_blacklist() -> dict:
    """Thread-safe read of scammer blacklist."""
    with _blacklist_lock:
        if not os.path.exists(BLACKLIST_PATH):
            return {}
        try:
            with open(BLACKLIST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning reading blacklist {BLACKLIST_PATH}: {e}")
            return {}

def mark_scammer(phone_number: str, risk_score: float, reason: str, metadata: dict = None) -> dict:
    """Atomically record flagged caller to scammer blacklist without data race."""
    info = lookup_carrier(phone_number)
    geo = get_ip_geolocation(metadata.get("ip") if metadata else None)

    caller_hash = hash_caller_identity(phone_number)
    clean_reason = reason.split("\n")[0][:100].strip() if reason else "Vishing / Banking Fraud"
    record = {
        "caller_hash": caller_hash,
        "phone_masked": mask_phone(phone_number),
        "carrier": info.get("carrier", "VoIP SIP Gateway"),
        "location": info.get("location", geo.get("city", "Unknown")),
        "lat": info.get("lat", geo.get("lat", 28.6139)),
        "lon": info.get("lon", geo.get("lon", 77.2090)),
        "risk_score": round(float(risk_score), 1),
        "reason": clean_reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": "BANKING_AND_UPI_LOCKED" if risk_score >= 70.0 else "FLAGGED_REVIEW"
    }

    with _blacklist_lock:
        data = {}
        if os.path.exists(BLACKLIST_PATH):
            try:
                with open(BLACKLIST_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[caller_hash] = record

        tmp_path = f"{BLACKLIST_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, BLACKLIST_PATH)

    return record

def is_blacklisted(phone_number: str):
    """Check if caller is marked as a scammer by checking canonical HMAC hash, legacy hash, or raw keys."""
    blacklist = load_blacklist()
    h_canonical = hash_caller_identity(phone_number)
    h_legacy = hash_caller_identity_legacy(phone_number)
    clean_phone = str(phone_number).strip()

    # Direct key lookup
    if h_canonical in blacklist:
        return True, blacklist[h_canonical]
    if h_legacy in blacklist:
        return True, blacklist[h_legacy]
    if clean_phone in blacklist:
        return True, blacklist[clean_phone]

    # Inner record caller_hash attribute lookup
    for key, entry in blacklist.items():
        rec_hash = entry.get("caller_hash")
        if rec_hash in (h_canonical, h_legacy):
            return True, entry

    return False, None

def main():
    parser = argparse.ArgumentParser(description="VaniShield AI Caller Tracker & Geolocation")
    parser.add_argument("--lookup", help="Lookup carrier and circle for a phone number")
    parser.add_argument("--mark", help="Mark a phone number as confirmed scammer")
    parser.add_argument("--risk", type=float, default=95.0, help="Scam risk score (0-100)")
    parser.add_argument("--reason", default="Vishing / Banking Fraud", help="Flag reason")
    parser.add_argument("--list", action="store_true", help="List all blacklisted scammers")

    args = parser.parse_args()

    if args.lookup:
        info = lookup_carrier(args.lookup)
        print("\n--- Caller Telecom Lookup ---")
        for k, v in info.items():
            print(f"  {k:<12}: {v}")
        is_bl, bl_info = is_blacklisted(args.lookup)
        print(f"  Blacklisted : {'YES (Risk ' + str(bl_info.get('risk_score')) + '%)' if is_bl else 'NO (Clear)'}\n")

    elif args.mark:
        rec = mark_scammer(args.mark, args.risk, args.reason)
        print(f"\n[OK] Caller {args.mark} successfully recorded to {BLACKLIST_PATH}")
        print(f"     Masked ID: {rec['phone_masked']} | Risk: {rec['risk_score']}% | Carrier: {rec['carrier']}\n")

    elif args.list:
        data = load_blacklist()
        print(f"\nActive Blacklisted Scammers ({len(data)} entries):")
        print("-" * 75)
        for num, info in data.items():
            print(f"{num:<16} | Risk: {info.get('risk_score', 0):>5.1f}% | {info.get('carrier', 'N/A'):<22} | {info.get('location', 'N/A')}")
        print("-" * 75 + "\n")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
