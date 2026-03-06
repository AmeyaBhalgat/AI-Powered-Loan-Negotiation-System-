
import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from enum import Enum

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_BACKEND    = "gemini"               # "gemini" or "llama"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
LLAMA_MODEL    = "llama3"
LLAMA_URL      = "http://localhost:11434/api/generate"

CONFIRM_THRESH   = 0.70
RECOUNTER_THRESH = 0.40
MAX_ROUNDS       = 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("LoanNeg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BorrowerProfile:
    credit_score: float
    annual_income: float
    debt_to_income: float
    employment_years: float
    loan_amount_requested: float
    loan_term_months: int
    collateral_value: float

@dataclass
class LoanOffer:
    interest_rate: float
    origination_fee: float
    loan_term_months: int
    monthly_payment: float
    total_repayment: float
    rationale: str = ""

class NegotiationStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    ACCEPTED    = "accepted"
    DECLINED    = "declined"
    FINALIZED   = "finalized"
    ESCALATED   = "escalated"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LANGGRAPH STATE
# This is the "shared memory" that every node reads from and writes to.
# LangGraph passes this dict between nodes automatically.
# ═══════════════════════════════════════════════════════════════════════════════

class LoanState(TypedDict):
    # ── Session info ──────────────────────────────────────────────────────────
    session_id:      str
    round_number:    int
    status:          str                    # NegotiationStatus value
    history:         list                   # full audit log

    # ── Borrower ──────────────────────────────────────────────────────────────
    borrower:        dict                   # BorrowerProfile as dict

    # ── XGBoost pre-screen results ────────────────────────────────────────────
    prescreen_prob:  Optional[float]
    recommended_range: Optional[tuple]
    rate_hint:       Optional[str]
    prescreen_explain: Optional[str]

    # ── Current system offer ──────────────────────────────────────────────────
    system_offer:    Optional[dict]         # LoanOffer as dict
    system_feasibility: Optional[float]
    system_risk:     Optional[str]
    system_explain:  Optional[str]

    # ── User action & counter ─────────────────────────────────────────────────
    user_action:     Optional[str]          # "accept" | "decline" | "counter"
    user_counter:    Optional[dict]         # LoanOffer as dict

    # ── User counter scoring ──────────────────────────────────────────────────
    counter_feasibility: Optional[float]
    counter_risk:    Optional[str]
    counter_explain: Optional[str]

    # ── Final outcome ─────────────────────────────────────────────────────────
    final_offer:     Optional[dict]
    suggestion:      Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — XGBOOST SCORER (singleton, shared across nodes)
# ═══════════════════════════════════════════════════════════════════════════════

class XGBoostScorer:
    def __init__(self):
        self.prescreener  = XGBClassifier(n_estimators=100, max_depth=4,
                                          use_label_encoder=False, eval_metric="logloss")
        self.offer_scorer = XGBClassifier(n_estimators=100, max_depth=5,
                                          use_label_encoder=False, eval_metric="logloss")
        self.scaler_pre   = StandardScaler()
        self.scaler_offer = StandardScaler()
        self._train_synthetic()

    def _train_synthetic(self, n=2000, seed=42):
        rng = np.random.default_rng(seed)
        cs  = rng.uniform(300, 850, n)
        inc = rng.uniform(25_000, 200_000, n)
        dti = rng.uniform(0.05, 0.65, n)
        emp = rng.uniform(0, 30, n)
        ltv = rng.uniform(0, 1.5, n)
        X_pre = np.column_stack([cs, inc, dti, emp, ltv])
        y_pre = ((cs > 650) & (dti < 0.45) & (inc > 40_000)).astype(int)
        self.prescreener.fit(self.scaler_pre.fit_transform(X_pre), y_pre)

        rate  = rng.uniform(3.0, 25.0, n)
        fee   = rng.uniform(0, 0.05, n)
        term  = rng.choice([24, 36, 48, 60, 84], n)
        X_off = np.column_stack([cs, inc, dti, emp, ltv, rate, fee, term])
        y_off = ((rate < (30 - cs / 50)) & (dti < 0.45)).astype(int)
        self.offer_scorer.fit(self.scaler_offer.fit_transform(X_off), y_off)
        log.info("XGBoost trained on %d synthetic samples.", n)

    def _ltv(self, b: dict) -> float:
        return b["loan_amount_requested"] / (b["collateral_value"] + 1e-9) \
               if b["collateral_value"] else 1.5

    def prescreen(self, b: dict) -> dict:
        X = self.scaler_pre.transform([[b["credit_score"], b["annual_income"],
                                        b["debt_to_income"], b["employment_years"],
                                        self._ltv(b)]])
        prob = float(self.prescreener.predict_proba(X)[0, 1])
        if prob >= 0.75:   rr, hint = (4.0, 8.0),   "prime"
        elif prob >= 0.50: rr, hint = (8.0, 14.0),  "standard"
        elif prob >= 0.30: rr, hint = (14.0, 20.0), "subprime"
        else:              rr, hint = (20.0, 28.0), "high-risk"
        explain = (f"Pre-screen score: {prob:.2f}. Credit={b['credit_score']}, "
                   f"DTI={b['debt_to_income']:.0%}, Income=${b['annual_income']:,.0f}. "
                   f"Recommended rate: {rr[0]}%–{rr[1]}%.")
        return {"prob": prob, "recommended_range": rr, "hint": hint, "explain": explain}

    def score_offer(self, b: dict, o: dict) -> dict:
        fee_pct = o["origination_fee"] / (b["loan_amount_requested"] + 1e-9)
        X = self.scaler_offer.transform([[b["credit_score"], b["annual_income"],
                                          b["debt_to_income"], b["employment_years"],
                                          self._ltv(b), o["interest_rate"],
                                          fee_pct, o["loan_term_months"]]])
        feas = float(self.offer_scorer.predict_proba(X)[0, 1])
        risk = "LOW" if feas >= 0.7 else "MEDIUM" if feas >= 0.4 else "HIGH"
        explain = (f"Feasibility: {feas:.2f} ({risk} risk). "
                   f"Rate={o['interest_rate']}%, Fee=${o['origination_fee']:,.0f}, "
                   f"Term={o['loan_term_months']}mo, Monthly=${o['monthly_payment']:,.2f}.")
        return {"feasibility": feas, "risk": risk, "explain": explain}


# Global scorer instance (shared by all nodes)
SCORER = XGBoostScorer()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LLM BACKEND
# ═══════════════════════════════════════════════════════════════════════════════

def call_llm(prompt: str) -> str:
    if LLM_BACKEND == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt).text.strip()
    elif LLM_BACKEND == "llama":
        import requests
        r = requests.post(LLAMA_URL,
                          json={"model": LLAMA_MODEL, "prompt": prompt, "stream": False},
                          timeout=120)
        r.raise_for_status()
        return r.json()["response"].strip()
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}")

def _parse_offer_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}") + 1
    if s == -1: raise ValueError("No JSON in LLM response")
    return json.loads(text[s:e])

def _compute_monthly(principal: float, annual_rate: float, months: int) -> float:
    r = annual_rate / 100 / 12
    if r == 0: return principal / months
    return principal * r * (1 + r)**months / ((1 + r)**months - 1)

def _build_offer(b: dict, data: dict) -> dict:
    monthly = _compute_monthly(b["loan_amount_requested"], data["interest_rate"], data["loan_term_months"])
    return {
        "interest_rate":    data["interest_rate"],
        "origination_fee":  data["origination_fee"],
        "loan_term_months": data["loan_term_months"],
        "monthly_payment":  monthly,
        "total_repayment":  monthly * data["loan_term_months"] + data["origination_fee"],
        "rationale":        data.get("rationale", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LANGGRAPH NODES
# Each node is a plain Python function: (LoanState) → dict of state updates
# LangGraph merges the returned dict back into the state automatically.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Node 1: Pre-screen borrower ───────────────────────────────────────────────
def prescreen_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: Runs XGBoost on borrower profile to determine risk level
                  and recommended interest rate range.
    INPUTS:  state["borrower"]
    OUTPUTS: prescreen_prob, recommended_range, rate_hint, prescreen_explain
    """
    log.info("NODE: prescreen_node")
    result = SCORER.prescreen(state["borrower"])
    entry  = {"step": "prescreen", **result}

    return {
        "prescreen_prob":    result["prob"],
        "recommended_range": result["recommended_range"],
        "rate_hint":         result["hint"],
        "prescreen_explain": result["explain"],
        "history":           state["history"] + [entry],
    }


# ── Node 2: Seller LLM generates initial offer ────────────────────────────────
def seller_offer_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: Calls LLM (Gemini/LLaMA) acting as the bank's loan officer.
                  LLM receives borrower data + XGBoost rate hint and outputs
                  a structured loan offer as JSON.
    INPUTS:  state["borrower"], state["recommended_range"], state["rate_hint"]
    OUTPUTS: system_offer (dict)
    """
    log.info("NODE: seller_offer_node")
    b = state["borrower"]

    prompt = f"""
You are a loan officer AI for a bank. Generate a loan offer for this borrower.
Return ONLY valid JSON — no markdown, no explanation outside the JSON.

Borrower Profile:
{json.dumps(b, indent=2)}

XGBoost Pre-screen:
- Recommended rate range: {state['recommended_range']}
- Risk category hint: {state['rate_hint']}
- Explanation: {state['prescreen_explain']}

Generate a competitive but profitable offer. JSON schema:
{{
  "interest_rate": <float, annual %>,
  "origination_fee": <float, USD>,
  "loan_term_months": <int>,
  "rationale": "<short string>"
}}
"""
    raw  = call_llm(prompt)
    log.debug("LLM response: %s", raw)
    data  = _parse_offer_json(raw)
    offer = _build_offer(b, data)

    return {
        "system_offer": offer,
        "history": state["history"] + [{"step": "seller_initial_offer", "offer": offer}],
    }


# ── Node 3: XGBoost scores the system offer ───────────────────────────────────
def score_system_offer_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: Validates the LLM-generated offer using XGBoost.
                  Ensures the AI didn't produce something unrealistic.
    INPUTS:  state["borrower"], state["system_offer"]
    OUTPUTS: system_feasibility, system_risk, system_explain
    """
    log.info("NODE: score_system_offer_node")
    result = SCORER.score_offer(state["borrower"], state["system_offer"])
    return {
        "system_feasibility": result["feasibility"],
        "system_risk":        result["risk"],
        "system_explain":     result["explain"],
        "history": state["history"] + [{"step": "score_system_offer", **result}],
    }


# ── Node 4: Present offer to user (Human-in-the-Loop) ────────────────────────
def present_to_user_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: Prints the current system offer to the user and waits
                  for their input. This is the Human-in-the-Loop checkpoint —
                  LangGraph PAUSES execution here and resumes when the user
                  provides their action via graph.update_state().
    INPUTS:  state["system_offer"], state["system_feasibility"]
    OUTPUTS: user_action (set externally after interrupt)
    """
    log.info("NODE: present_to_user_node — waiting for user input")
    o = state["system_offer"]

    print(f"\n{'═'*57}")
    print(f"  LOAN OFFER  (Round {state['round_number'] + 1})")
    print(f"{'─'*57}")
    print(f"  Interest Rate    : {o['interest_rate']:.2f}%")
    print(f"  Origination Fee  : ${o['origination_fee']:,.2f}")
    print(f"  Term             : {o['loan_term_months']} months")
    print(f"  Monthly Payment  : ${o['monthly_payment']:,.2f}")
    print(f"  Total Repayment  : ${o['total_repayment']:,.2f}")
    print(f"  Risk Level       : {state['system_risk']}  (feasibility: {state['system_feasibility']:.2f})")
    print(f"  Rationale        : {o['rationale']}")
    print(f"  XGB Analysis     : {state['system_explain']}")
    print(f"{'═'*57}")

    # Collect user input inline (in production this would be an API interrupt)
    print("\n[Your options]  A) Accept   B) Decline   C) Counter-offer")
    choice = input("Choice: ").strip().lower()

    if choice == "a":
        return {"user_action": "accept", "round_number": state["round_number"] + 1}
    elif choice == "b":
        return {"user_action": "decline", "round_number": state["round_number"] + 1}
    elif choice == "c":
        print("\n[Enter your counter-offer]")
        rate = float(input("  Interest rate (%): "))
        fee  = float(input("  Origination fee ($): "))
        term = int(input("  Term (months): "))
        b    = state["borrower"]
        monthly = _compute_monthly(b["loan_amount_requested"], rate, term)
        counter = {
            "interest_rate": rate, "origination_fee": fee,
            "loan_term_months": term, "monthly_payment": monthly,
            "total_repayment": monthly * term + fee, "rationale": "user counter",
        }
        return {
            "user_action":  "counter",
            "user_counter": counter,
            "round_number": state["round_number"] + 1,
        }
    else:
        print("  Invalid input. Defaulting to decline.")
        return {"user_action": "decline"}


# ── Node 5: Accept → finalize ─────────────────────────────────────────────────
def accept_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: User accepted the system offer. Marks the loan as accepted.
    """
    log.info("NODE: accept_node")
    print(f"\n✅ Loan ACCEPTED! Final rate: {state['system_offer']['interest_rate']:.2f}%")
    return {
        "status":     NegotiationStatus.ACCEPTED,
        "final_offer": state["system_offer"],
        "history":    state["history"] + [{"step": "user_accepted"}],
    }


# ── Node 6: Decline → end ────────────────────────────────────────────────────
def decline_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: User declined the offer. Ends the negotiation.
    """
    log.info("NODE: decline_node")
    print("\n❌ Loan DECLINED. Session ended.")
    return {
        "status":  NegotiationStatus.DECLINED,
        "history": state["history"] + [{"step": "user_declined"}],
    }


# ── Node 7: Score the user's counter-offer ────────────────────────────────────
def score_counter_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: XGBoost evaluates the borrower's counter-offer.
                  The feasibility score determines what happens next.
    INPUTS:  state["borrower"], state["user_counter"]
    OUTPUTS: counter_feasibility, counter_risk, counter_explain
    """
    log.info("NODE: score_counter_node")
    result = SCORER.score_offer(state["borrower"], state["user_counter"])
    log.info("Counter feasibility=%.2f, risk=%s", result["feasibility"], result["risk"])
    return {
        "counter_feasibility": result["feasibility"],
        "counter_risk":        result["risk"],
        "counter_explain":     result["explain"],
        "history": state["history"] + [
            {"step": "user_counter", "offer": state["user_counter"], **result}
        ],
    }


# ── Node 8: Finalize user's counter ──────────────────────────────────────────
def finalize_counter_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: User's counter scored ≥ CONFIRM_THRESH → auto-accept it.
    """
    log.info("NODE: finalize_counter_node")
    fo = state["user_counter"]
    print(f"\n✅ Counter-offer ACCEPTED by the system!")
    print(f"   Final Rate: {fo['interest_rate']:.2f}%, Monthly: ${fo['monthly_payment']:,.2f}")
    return {
        "status":     NegotiationStatus.FINALIZED,
        "final_offer": fo,
        "history":    state["history"] + [{"step": "counter_finalized"}],
    }


# ── Node 9: Seller LLM re-counters ───────────────────────────────────────────
def recounter_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: Counter was in the RECOUNTER_THRESH–CONFIRM_THRESH range.
                  Calls the LLM to generate a revised offer that moves toward
                  the borrower's request while staying within bank constraints.
    INPUTS:  state["user_counter"], state["counter_feasibility"], state["borrower"]
    OUTPUTS: system_offer (updated revised offer)
    """
    log.info("NODE: recounter_node")
    b   = state["borrower"]
    min_rate = state["recommended_range"][0]

    prompt = f"""
You are a loan officer AI. The borrower countered with an offer slightly outside
our acceptable range. Generate a revised counter-offer that moves toward the
borrower's request while staying within bank constraints.
Return ONLY valid JSON — no markdown.

Borrower Profile:
{json.dumps(b, indent=2)}

Borrower's Counter-Offer:
{json.dumps(state['user_counter'], indent=2)}

XGBoost Feasibility of their counter: {state['counter_feasibility']:.2f} (we need ≥ {CONFIRM_THRESH})
Risk: {state['counter_risk']}
Explanation: {state['counter_explain']}

Constraints:
- interest_rate must be ≥ {min_rate}%
- origination_fee must be ≥ $500
- Move toward borrower's preferred rate and term as much as possible

JSON schema:
{{
  "interest_rate": <float>,
  "origination_fee": <float>,
  "loan_term_months": <int>,
  "rationale": "<short string explaining the compromise>"
}}
"""
    raw  = call_llm(prompt)
    data  = _parse_offer_json(raw)
    offer = _build_offer(b, data)

    print(f"\n🔄 Seller revised offer: {offer['interest_rate']:.2f}%, ${offer['origination_fee']:,.0f} fee")
    return {
        "system_offer": offer,
        "history": state["history"] + [{"step": "seller_recounter", "offer": offer}],
    }


# ── Node 10: Score the re-counter offer ──────────────────────────────────────
def score_recounter_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: XGBoost scores the seller's re-counter offer before
                  presenting it to the user again.
    """
    log.info("NODE: score_recounter_node")
    result = SCORER.score_offer(state["borrower"], state["system_offer"])
    return {
        "system_feasibility": result["feasibility"],
        "system_risk":        result["risk"],
        "system_explain":     result["explain"],
        "history": state["history"] + [{"step": "score_recounter", **result}],
    }


# ── Node 11: Escalate ────────────────────────────────────────────────────────
def escalate_node(state: LoanState) -> dict:
    """
    WHAT IT DOES: Counter was below RECOUNTER_THRESH or max rounds reached.
                  Ends negotiation with suggestions for the borrower.
    """
    log.info("NODE: escalate_node")
    b = state["borrower"]
    min_rate = state["recommended_range"][0]
    msgs = []
    uc   = state.get("user_counter") or {}
    if uc.get("interest_rate", 0) < min_rate:
        msgs.append(f"Consider raising your rate to at least {min_rate:.1f}%.")
    if uc.get("loan_term_months", 60) < 36:
        msgs.append("A longer term (≥ 36 months) may improve feasibility.")
    if uc.get("origination_fee", 1000) < 500:
        msgs.append("Origination fee below $500 is not viable.")
    suggestion = " ".join(msgs) or "Please contact a loan specialist for alternatives."

    print(f"\n❌ Negotiation ESCALATED. Counter could not be accommodated.")
    print(f"   Suggestion: {suggestion}")
    return {
        "status":     NegotiationStatus.ESCALATED,
        "suggestion": suggestion,
        "history":    state["history"] + [{"step": "escalated", "suggestion": suggestion}],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONDITIONAL EDGE FUNCTIONS
# These are the "routing" functions — they look at state and decide which
# node to go to next. This is where LangGraph shines vs plain Python.
# ═══════════════════════════════════════════════════════════════════════════════

def route_user_action(state: LoanState) -> Literal["accept_node", "decline_node", "score_counter_node"]:
    """Routes based on what the user chose: accept / decline / counter."""
    action = state.get("user_action", "decline")
    if action == "accept":  return "accept_node"
    if action == "decline": return "decline_node"
    return "score_counter_node"


def route_counter_result(state: LoanState) -> Literal["finalize_counter_node", "recounter_node", "escalate_node"]:
    """
    Routes based on XGBoost feasibility of the user's counter:
      ≥ 0.70 → finalize (auto-accept)
      0.40–0.70 AND rounds left → recounter (LLM re-negotiates)
      else → escalate
    """
    feas   = state.get("counter_feasibility", 0)
    rounds = state.get("round_number", 0)

    if feas >= CONFIRM_THRESH:
        return "finalize_counter_node"
    if feas >= RECOUNTER_THRESH and rounds < MAX_ROUNDS:
        return "recounter_node"
    return "escalate_node"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUILD THE LANGGRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    Constructs and compiles the LangGraph state machine.
    
    Graph structure:
    
      prescreen_node
           ↓
      seller_offer_node
           ↓
      score_system_offer_node
           ↓
      present_to_user_node ──────────────────────────┐
           ↓                                          │ (loop back)
      [route_user_action]                             │
       ↙      ↓       ↘                              │
    accept  decline  score_counter_node               │
      ↓       ↓           ↓                          │
     END     END    [route_counter_result]            │
                     ↙       ↓       ↘               │
               finalize  recounter  escalate          │
                  ↓          ↓         ↓             │
                 END   score_recounter  END           │
                             ↓                        │
                       present_to_user ───────────────┘
    """
    builder = StateGraph(LoanState)

    # ── Add all nodes ─────────────────────────────────────────────────────────
    builder.add_node("prescreen_node",          prescreen_node)
    builder.add_node("seller_offer_node",       seller_offer_node)
    builder.add_node("score_system_offer_node", score_system_offer_node)
    builder.add_node("present_to_user_node",    present_to_user_node)
    builder.add_node("accept_node",             accept_node)
    builder.add_node("decline_node",            decline_node)
    builder.add_node("score_counter_node",      score_counter_node)
    builder.add_node("finalize_counter_node",   finalize_counter_node)
    builder.add_node("recounter_node",          recounter_node)
    builder.add_node("score_recounter_node",    score_recounter_node)
    builder.add_node("escalate_node",           escalate_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("prescreen_node")

    # ── Linear edges (A always goes to B) ────────────────────────────────────
    builder.add_edge("prescreen_node",          "seller_offer_node")
    builder.add_edge("seller_offer_node",       "score_system_offer_node")
    builder.add_edge("score_system_offer_node", "present_to_user_node")

    # ── Conditional edge: after user responds ─────────────────────────────────
    builder.add_conditional_edges(
        "present_to_user_node",
        route_user_action,
        {
            "accept_node":        "accept_node",
            "decline_node":       "decline_node",
            "score_counter_node": "score_counter_node",
        }
    )

    # ── Conditional edge: after scoring user's counter ────────────────────────
    builder.add_conditional_edges(
        "score_counter_node",
        route_counter_result,
        {
            "finalize_counter_node": "finalize_counter_node",
            "recounter_node":        "recounter_node",
            "escalate_node":         "escalate_node",
        }
    )

    # ── After re-counter: score it, then loop back to present_to_user ────────
    builder.add_edge("recounter_node",       "score_recounter_node")
    builder.add_edge("score_recounter_node", "present_to_user_node")

    # ── Terminal edges ────────────────────────────────────────────────────────
    builder.add_edge("accept_node",           END)
    builder.add_edge("decline_node",          END)
    builder.add_edge("finalize_counter_node", END)
    builder.add_edge("escalate_node",         END)

    # ── Compile with in-memory checkpointing (persists state between steps) ───
    memory = MemorySaver()
    graph  = builder.compile(checkpointer=memory)

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_negotiation(borrower_dict: dict):
    """
    Initializes LoanState and runs the LangGraph until END.
    Each session gets a unique thread_id for checkpointing.
    """
    graph      = build_graph()
    session_id = str(uuid.uuid4())
    thread_id  = session_id

    print(f"\n{'═'*57}")
    print(f"  LOAN NEGOTIATION SYSTEM  (LangGraph Edition)")
    print(f"  Session: {session_id[:8]}...")
    print(f"  Loan Request: ${borrower_dict['loan_amount_requested']:,.0f}  "
          f"| {borrower_dict['loan_term_months']} months")
    print(f"{'═'*57}")

    # ── Initial state ─────────────────────────────────────────────────────────
    initial_state: LoanState = {
        "session_id":          session_id,
        "round_number":        0,
        "status":              NegotiationStatus.IN_PROGRESS,
        "history":             [],
        "borrower":            borrower_dict,
        "prescreen_prob":      None,
        "recommended_range":   None,
        "rate_hint":           None,
        "prescreen_explain":   None,
        "system_offer":        None,
        "system_feasibility":  None,
        "system_risk":         None,
        "system_explain":      None,
        "user_action":         None,
        "user_counter":        None,
        "counter_feasibility": None,
        "counter_risk":        None,
        "counter_explain":     None,
        "final_offer":         None,
        "suggestion":          None,
    }

    # ── Run the graph (streaming so we see node-by-node progress) ─────────────
    config = {"configurable": {"thread_id": thread_id}}
    final_state = None

    for step in graph.stream(initial_state, config=config):
        node_name = list(step.keys())[0]
        log.info("✓ Completed node: %s", node_name)
        final_state = step[node_name]

    # ── Summary ───────────────────────────────────────────────────────────────
    if final_state:
        print(f"\n{'─'*57}")
        print(f"  NEGOTIATION COMPLETE")
        print(f"  Status       : {final_state.get('status', 'unknown')}")
        print(f"  Rounds taken : {final_state.get('round_number', 0)}")
        fo = final_state.get("final_offer")
        if fo:
            print(f"  Final Rate   : {fo['interest_rate']:.2f}%")
            print(f"  Monthly Pmt  : ${fo['monthly_payment']:,.2f}")
            print(f"  Total Cost   : ${fo['total_repayment']:,.2f}")
        print(f"  Audit log    : {len(final_state.get('history', []))} entries")
        print(f"{'─'*57}")


def run_demo():
    """Quick demo with pre-filled borrower."""
    borrower = {
        "credit_score":           680,
        "annual_income":          75_000,
        "debt_to_income":         0.32,
        "employment_years":       5,
        "loan_amount_requested":  25_000,
        "loan_term_months":       60,
        "collateral_value":       0,
    }
    run_negotiation(borrower)


def run_interactive():
    """Collect borrower info from terminal."""
    print("\n═══════════  LOAN NEGOTIATION  ═══════════")
    borrower = {
        "credit_score":          float(input("Credit score (300-850): ")),
        "annual_income":         float(input("Annual income ($): ")),
        "debt_to_income":        float(input("Debt-to-income ratio (e.g. 0.30): ")),
        "employment_years":      float(input("Years employed: ")),
        "loan_amount_requested": float(input("Loan amount requested ($): ")),
        "loan_term_months":      int(input("Desired term (months): ")),
        "collateral_value":      float(input("Collateral value (0 if none): ")),
    }
    run_negotiation(borrower)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        run_interactive()