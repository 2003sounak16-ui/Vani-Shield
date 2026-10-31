"""
VaniShield AI — Banking Hold & Circuit Breaker API
Team CypherEye | Project VaniShield AI
Provides automated and manual payment freeze endpoints for UPI and Net Banking.
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/banking", tags=["Banking Circuit Breaker"])

class BankingState:
    def __init__(self):
        self.locked = False
        self.risk_score = 0.0
        self.reason = "Account Normal"
        self.locked_at = None
        self.account_masked = "HDFC-****-****-4921"
        self.upi_id = "user@okhdfcbank"

state = BankingState()

class LockRequest(BaseModel):
    reason: str
    risk_score: float = 85.0

import hmac
import os
CONFIGURED_PIN = os.environ.get("VANISHIELD_UNLOCK_PIN", "1234")

class UnlockRequest(BaseModel):
    pin: str
    reason: str = "Biometrically authorized user release"

class TransferRequest(BaseModel):
    payee: str
    amount: float
    pin: str

@router.get("/status")
async def get_banking_status():
    """Retrieve current banking security and circuit-breaker status."""
    return {
        "status": "LOCKED" if state.locked else "ACTIVE",
        "locked": state.locked,
        "risk_score": state.risk_score,
        "reason": state.reason,
        "locked_at": state.locked_at,
        "account_masked": state.account_masked,
        "upi_id": state.upi_id,
        "protection": "VaniShield Biometric Interlock Active"
    }

@router.post("/lock")
async def lock_banking(req: LockRequest):
    """Trigger emergency banking hold circuit breaker."""
    state.locked = True
    state.risk_score = req.risk_score
    state.reason = req.reason
    state.locked_at = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "success": True,
        "status": "LOCKED",
        "message": f"Emergency circuit breaker engaged. UPI & Net Banking disabled: {req.reason}",
        "locked_at": state.locked_at
    }

@router.post("/unlock")
async def unlock_banking(req: UnlockRequest):
    """Release banking hold via PIN verification."""
    if not req.pin or not hmac.compare_digest(req.pin.strip(), CONFIGURED_PIN):
        raise HTTPException(status_code=403, detail="Invalid security PIN.")
    state.locked = False
    state.risk_score = 0.0
    state.reason = req.reason or "Biometrically authorized user release"
    state.locked_at = None
    return {
        "success": True,
        "status": "ACTIVE",
        "message": "Banking hold released. Normal UPI transactions restored."
    }

@router.post("/simulate-transfer")
async def simulate_transfer(req: TransferRequest):
    """Attempt a simulated UPI transfer — automatically blocked if circuit breaker is locked."""
    if state.locked:
        raise HTTPException(
            status_code=423,
            detail=f"TRANSACTION BLOCKED: VaniShield Automated Circuit Breaker is active! ({state.reason})"
        )
    return {
        "success": True,
        "message": f"Successfully transferred Rs. {req.amount:,.2f} to {req.payee}.",
        "tx_id": f"UPI{int(time.time()*1000)}"
    }
