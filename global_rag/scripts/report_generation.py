# File_name: report_generation.py
# Purpose:
# Generate a SOW-driven AI first-line IC review pack for investment submissions.

# This is NOT a generic market research report generator.

# The script:
# 1. Uses retrieve_chunks.py for staged evidence retrieval
# 2. Builds SOW-aligned review modules
# 3. Clearly marks unavailable SOW areas as Grey / Not assessed
# 4. Produces JSON and Markdown outputs
# 5. Optionally writes an audit record to PostgreSQL

# Assumptions:
# - global_rag.scripts.config contains config_base() and config_paths()
# - global_rag.scripts.retrieve_chunks contains retrieve_chunks()
# - chunks table and embeddings are already populated

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from openai import OpenAI

from global_rag.scripts import config
import global_rag.scripts.retrieve_chunks as ret


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REPORT_GENERATION_VERSION = "report_generation_poc_v1"
REPORT_TYPE = "ai_first_line_ic_review"
CLASSIFICATION = "Confidential External"

STATUS_GREEN = "Green"
STATUS_AMBER = "Amber"
STATUS_RED = "Red"
STATUS_GREY = "Grey"

READINESS_PASS = "Pass"
READINESS_PARTIAL = "Partial"
READINESS_MISSING = "Missing"
READINESS_WEAK = "Weak"
READINESS_NOT_ASSESSED = "Not assessed"


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\x00", " ")
    value = value.replace("\r", " ")
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_text(value):
    return clean_text(value).lower()


def safe_float(value):
    if value is None:
        return None

    try:
        value = str(value)
        value = value.replace(",", "")
        value = value.replace("%", "")
        value = value.replace("x", "")
        value = value.strip()
        return float(value)
    except Exception:
        return None


def has_any(text_value, terms):
    text_value = normalize_text(text_value)

    for term in terms:
        if normalize_text(term) in text_value:
            return True

    return False


def has_all(text_value, terms):
    text_value = normalize_text(text_value)

    for term in terms:
        if normalize_text(term) not in text_value:
            return False

    return True


def make_run_id(transaction_id):
    raw_value = f"{transaction_id}_{utc_now_iso()}_{REPORT_GENERATION_VERSION}"
    digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:12]
    return f"RPT_{transaction_id}_{digest}"


def get_config_pack():
    config_base = config.config_base()
    config_paths = config.config_paths()

    draft_report_dir = Path(config_paths["draft_report_dir"])
    draft_report_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config_base": config_base,
        "config_paths": config_paths,
        "draft_report_dir": draft_report_dir,
    }


def get_evidence_blob(results):
    text_parts = []

    for item in results:
        text_parts.append(clean_text(item.get("chunk_text")))
        text_parts.append(clean_text(item.get("section_heading")))
        text_parts.append(clean_text(item.get("source_reference")))
        text_parts.append(clean_text(item.get("document_id")))
        text_parts.append(clean_text(item.get("corpus_zone")))
        text_parts.append(clean_text(item.get("corpus_pack")))

    return " ".join(text_parts)


def get_source_reference(result):
    source_reference = clean_text(result.get("source_reference"))

    table_id = None
    table_match = re.search(r"table_id=([^;]+)", source_reference)
    if table_match:
        table_id = table_match.group(1).strip()

    page_no = None
    page_match = re.search(r"page_no=([^;]+)", source_reference)
    if page_match:
        page_no = page_match.group(1).strip()

    row_start = None
    row_end = None

    row_start_match = re.search(r"row_start=([^;]+)", source_reference)
    row_end_match = re.search(r"row_end=([^;]+)", source_reference)

    if row_start_match:
        row_start = row_start_match.group(1).strip()

    if row_end_match:
        row_end = row_end_match.group(1).strip()

    return {
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id"),
        "corpus_zone": result.get("corpus_zone"),
        "corpus_pack": result.get("corpus_pack"),
        "section_heading": result.get("section_heading"),
        "page_start": result.get("page_start"),
        "page_end": result.get("page_end"),
        "page_no": page_no,
        "table_id": table_id,
        "row_start": row_start,
        "row_end": row_end,
        "source_reference": source_reference,
    }


def find_source_chunks(results, terms, max_sources=5):
    matched_sources = []

    for item in results:
        chunk_text = clean_text(item.get("chunk_text"))

        if has_any(chunk_text, terms):
            matched_sources.append(get_source_reference(item))

        if len(matched_sources) >= max_sources:
            break

    return matched_sources


def short_evidence_text(results, terms, max_chars=600):
    for item in results:
        chunk_text = clean_text(item.get("chunk_text"))

        if has_any(chunk_text, terms):
            return chunk_text[:max_chars]

    return ""


def dedupe_results(results):
    seen_signatures = set()
    deduped = []

    for item in results:
        chunk_text = normalize_text(item.get("chunk_text"))
        signature_text = chunk_text[:900]
        signature_text = re.sub(r"\s+", " ", signature_text)
        signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()

        if signature in seen_signatures:
            continue

        seen_signatures.add(signature)
        deduped.append(item)

    return deduped


def classify_evidence_type(result):
    corpus_zone = clean_text(result.get("corpus_zone"))
    corpus_pack = clean_text(result.get("corpus_pack"))
    document_id = clean_text(result.get("document_id"))
    section_heading = clean_text(result.get("section_heading"))
    source_reference = clean_text(result.get("source_reference"))
    chunk_text = clean_text(result.get("chunk_text"))

    combined = normalize_text(
        f"{corpus_zone} {corpus_pack} {document_id} {section_heading} {source_reference} {chunk_text}"
    )

    if corpus_zone == "client_data":
        if "workbook" in combined or "financial_assumptions" in combined or "doc_000016" in combined:
            return "client_workbook_control"
        if "investment committee memorandum" in combined or "doc_000014" in combined:
            return "client_memo"
        if "slide_" in combined or "ppt" in combined or "doc_000008" in combined:
            return "client_deck"
        if "page_" in combined or "pdf" in combined or "doc_000001" in combined:
            return "client_pdf"
        return "client_evidence"

    if corpus_zone == "corpus_data":
        if "benchmark" in combined or "nrel" in combined or "utility-scale" in combined:
            return "external_benchmark"
        if "seed" in combined or "risk" in combined:
            return "risk_taxonomy"
        return "public_context"

    return "unknown"


def score_result_for_module(result, module_key):
    score = 0.0

    similarity_score = result.get("similarity_score")
    hybrid_score = result.get("hybrid_score")
    keyword_score = result.get("keyword_score")

    if isinstance(similarity_score, (float, int)):
        score += float(similarity_score) * 10

    if isinstance(hybrid_score, (float, int)):
        score += float(hybrid_score) * 100

    if isinstance(keyword_score, (float, int)):
        score += float(keyword_score)

    section_heading = normalize_text(result.get("section_heading"))
    chunk_text = normalize_text(result.get("chunk_text"))
    evidence_type = classify_evidence_type(result)
    token_count = result.get("token_count_estimate") or 0

    if section_heading.startswith("table:"):
        score += 2.5

    if evidence_type == "client_workbook_control":
        if module_key in [
            "financial_reconciliation",
            "market_offtake_revenue",
            "sensitivity_review",
            "client_valuation",
        ]:
            score += 3.0

    if evidence_type in ["client_memo", "client_deck", "client_pdf", "client_evidence"]:
        if module_key in [
            "completeness_readiness",
            "strategy_fit",
            "risk_review",
            "conditions_open_items",
        ]:
            score += 1.5

    if evidence_type == "external_benchmark":
        if module_key in ["external_cost_benchmark", "external_ev_ebitda_benchmark"]:
            score += 4.0

    if token_count < 80 and not section_heading.startswith("table:"):
        score -= 3.0

    module_boost_terms = {
        "risk_review": [
            "risk",
            "rating",
            "mitigation",
            "merchant exposure",
            "grid cost",
            "bess degradation",
            "offtaker credit",
            "cybersecurity",
        ],
        "conditions_open_items": [
            "conditions",
            "approve progression",
            "final documentation",
            "metric reconciliation",
            "merchant-price",
            "offtaker-credit",
            "decommissioning",
        ],
        "financial_reconciliation": [
            "project irr",
            "equity irr",
            "npv",
            "minimum dscr",
            "ev / ebitda",
            "total project cost",
            "metrics differ",
        ],
        "sensitivity_review": [
            "sensitivity",
            "combined downside",
            "bess degradation",
            "augmentation",
            "equity irr",
            "minimum dscr",
            "included_in_submission",
        ],
        "market_offtake_revenue": [
            "contracted share",
            "merchant share",
            "ppa price",
            "merchant price",
            "bess revenue",
            "payment security",
            "offtaker",
        ],
        "external_cost_benchmark": [
            "utility-scale pv-plus-battery",
            "utility-scale battery storage",
            "solar - utility pv",
            "overnight capital cost",
            "grid connection cost",
            "fixed operating expenses",
        ],
    }

    for term in module_boost_terms.get(module_key, []):
        if term in chunk_text:
            score += 0.75

    return score


def select_evidence(results, module_key, top_n=10):
    results = dedupe_results(results)

    scored_results = []

    for item in results:
        scored_results.append(
            {
                "score": score_result_for_module(item, module_key),
                "item": item,
            }
        )

    scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)

    selected = []
    for row in scored_results[:top_n]:
        selected.append(row["item"])

    return selected


# ---------------------------------------------------------------------
# Config objects
# ---------------------------------------------------------------------

def load_review_config():
    completeness_checklist = [
        {
            "item_id": "C01_APPROVAL_REQUEST",
            "required_item": "Clear approval request and transaction overview",
            "evidence_terms": ["approval", "approve progression", "requested approval", "transaction overview"],
            "weak_terms": [],
        },
        {
            "item_id": "C02_FINANCIAL_METRICS",
            "required_item": "Core financial metrics and return case",
            "evidence_terms": ["project irr", "equity irr", "npv", "minimum dscr", "ev / ebitda"],
            "weak_terms": ["metrics differ", "different values", "reconciliation"],
        },
        {
            "item_id": "C03_REVENUE_OFFTAKE",
            "required_item": "Revenue, offtake and merchant exposure assumptions",
            "evidence_terms": ["contracted share", "merchant share", "ppa price", "merchant price", "bess revenue"],
            "weak_terms": ["not appended", "under negotiation", "no independent", "referenced but not included"],
        },
        {
            "item_id": "C04_SENSITIVITY_ANALYSIS",
            "required_item": "Sensitivity analysis and downside cases",
            "evidence_terms": ["sensitivity", "merchant price -15", "generation -5", "capex +10", "combined downside"],
            "weak_terms": ["not shown", "not included", "absent from submitted documents"],
        },
        {
            "item_id": "C05_RISK_REGISTER",
            "required_item": "Risk register / principal risks and mitigants",
            "evidence_terms": ["principal risks", "risk", "mitigation", "mitigants", "rating"],
            "weak_terms": ["not assessed", "under negotiation", "absent", "not modelled"],
        },
        {
            "item_id": "C06_MERCHANT_CURVE_SUPPORT",
            "required_item": "Independent merchant-price curve support",
            "evidence_terms": ["merchant curve", "merchant-price", "merchant price"],
            "weak_terms": ["not appended", "not included", "missing", "independent curve absent"],
        },
        {
            "item_id": "C07_OFFTAKER_CREDIT",
            "required_item": "Offtaker credit and payment-security evidence",
            "evidence_terms": ["offtaker", "credit", "payment security", "letter of credit"],
            "weak_terms": ["under discussion", "under negotiation", "not executed", "no independent credit"],
        },
        {
            "item_id": "C08_ESG_LEGAL_REGULATORY",
            "required_item": "ESG, legal and regulatory review",
            "evidence_terms": ["esg", "legal", "regulatory", "environmental", "social"],
            "weak_terms": ["open items", "remain open", "updates", "not assessed"],
        },
        {
            "item_id": "C09_LAND_PERMITTING",
            "required_item": "Land, easements and permits",
            "evidence_terms": ["land", "easement", "permit", "permitting"],
            "weak_terms": ["open", "remain open", "final easements", "updates"],
        },
        {
            "item_id": "C10_CYBERSECURITY",
            "required_item": "Cybersecurity / EMS / SCADA assessment",
            "evidence_terms": ["cyber", "cybersecurity", "ems", "scada"],
            "weak_terms": ["not assessed", "no dedicated assessment", "no dedicated ems/scada"],
        },
        {
            "item_id": "C11_DECOMMISSIONING",
            "required_item": "Decommissioning and battery-disposal reserve",
            "evidence_terms": ["decommissioning", "battery-disposal", "battery disposal"],
            "weak_terms": ["not assessed", "no funded", "no reserve"],
        },
        {
            "item_id": "C12_BENCHMARK_SUPPORT",
            "required_item": "Benchmark support for costs / valuation",
            "evidence_terms": ["benchmark", "upper quartile", "high", "cost per mw", "overnight capital cost"],
            "weak_terms": ["client-provided", "not independent", "not available", "not evidenced"],
        },
        {
            "item_id": "C13_CONDITIONS_PRECEDENT",
            "required_item": "Conditions precedent / open items before final approval",
            "evidence_terms": ["conditions", "open items", "final documentation", "final approval", "approve progression"],
            "weak_terms": ["subject to", "require", "close", "obtain", "reconcile"],
        },
    ]

    strategy_criteria = [
        {
            "criterion_id": "S01_ENERGY_SCOPE",
            "criterion": "Investment is within energy-sector / renewable generation scope",
            "configured": True,
            "positive_terms": ["solar", "pv", "bess", "storage", "renewable"],
            "negative_terms": [],
        },
        {
            "criterion_id": "S02_TECHNOLOGY_SCOPE",
            "criterion": "Technology matches target technologies",
            "configured": True,
            "positive_terms": ["solar", "pv", "bess", "storage"],
            "negative_terms": [],
        },
        {
            "criterion_id": "S03_GEOGRAPHY_PRIORITY",
            "criterion": "Geography is within configured priority geographies",
            "configured": False,
            "positive_terms": [],
            "negative_terms": [],
        },
        {
            "criterion_id": "S04_CONTROL_RIGHTS",
            "criterion": "Ownership / governance structure is aligned with control criteria",
            "configured": False,
            "positive_terms": ["80%", "controlling", "control rights", "directors"],
            "negative_terms": [],
        },
        {
            "criterion_id": "S05_PORTFOLIO_CONCENTRATION",
            "criterion": "Portfolio concentration by geography/counterparty is assessed",
            "configured": True,
            "positive_terms": ["portfolio concentration", "counterparty concentration", "geography", "counterparty"],
            "negative_terms": ["does not quantify", "not quantify", "not assessed"],
        },
        {
            "criterion_id": "S06_RETURN_HURDLE",
            "criterion": "Return hurdle / capital allocation threshold is evidenced",
            "configured": False,
            "positive_terms": ["hurdle", "return hurdle", "capital allocation"],
            "negative_terms": [],
        },
    ]

    common_ic_questions = [
        {
            "question_id": "ICQ01",
            "theme": "Merchant exposure",
            "question": "Is the 30% merchant exposure supported by an independent merchant-price curve and downside confidence range?",
            "coverage_terms": ["merchant exposure", "merchant curve", "merchant price"],
            "weak_terms": ["not appended", "independent curve absent", "not included", "missing"],
        },
        {
            "question_id": "ICQ02",
            "theme": "Offtaker credit",
            "question": "Has the offtaker's creditworthiness and payment-security package been independently assessed?",
            "coverage_terms": ["offtaker", "credit", "payment security", "letter of credit"],
            "weak_terms": ["under discussion", "under negotiation", "not executed", "no independent"],
        },
        {
            "question_id": "ICQ03",
            "theme": "Downside protection",
            "question": "Has the downside case combined merchant price, generation, capex, COD delay and FX sensitivities?",
            "coverage_terms": ["combined downside", "sensitivity", "merchant price", "generation", "capex", "cod delay", "fx"],
            "weak_terms": ["not included", "not shown", "absent"],
        },
        {
            "question_id": "ICQ04",
            "theme": "BESS augmentation",
            "question": "Is BESS degradation / augmentation cost explicitly modelled and supported by warranty / supplier evidence?",
            "coverage_terms": ["bess degradation", "augmentation", "warranty"],
            "weak_terms": ["not separately modelled", "not included", "not shown", "term sheet"],
        },
        {
            "question_id": "ICQ05",
            "theme": "Grid cost and schedule",
            "question": "Is grid interconnection cost and schedule confirmed by the utility, and is contingency adequate?",
            "coverage_terms": ["grid", "interconnection", "utility", "schedule", "contingency"],
            "weak_terms": ["ongoing", "under review", "confirmation", "not confirmed"],
        },
        {
            "question_id": "ICQ06",
            "theme": "IRR bridge",
            "question": "Is there an IRR bridge explaining key value drivers versus seller / management assumptions?",
            "coverage_terms": ["irr bridge", "seller", "management assumptions", "business case"],
            "weak_terms": ["not included", "missing", "not evidenced"],
        },
        {
            "question_id": "ICQ07",
            "theme": "Accounting impact / book value",
            "question": "Are accounting impact, book values and consolidation implications addressed?",
            "coverage_terms": ["accounting", "book value", "consolidation"],
            "weak_terms": ["not included", "missing", "not evidenced"],
        },
    ]

    asset_class_risk_taxonomy = [
        {
            "risk_id": "R01_MERCHANT_EXPOSURE",
            "risk": "Merchant price exposure",
            "asset_class": "Solar / BESS",
            "terms": ["merchant exposure", "merchant price exposure", "merchant price", "merchant curve"],
            "reviewer_challenge": "Has merchant revenue been supported by an independent price curve and downside confidence range?",
        },
        {
            "risk_id": "R02_GRID_COST_SCHEDULE",
            "risk": "Grid cost and schedule",
            "asset_class": "Solar / BESS",
            "terms": ["grid cost", "grid interconnection", "grid connection", "utility design"],
            "reviewer_challenge": "Is the grid interconnection scope, cost and schedule confirmed by the utility?",
        },
        {
            "risk_id": "R03_BESS_DEGRADATION",
            "risk": "BESS degradation / augmentation",
            "asset_class": "BESS",
            "terms": ["bess degradation", "battery augmentation", "bess augmentation", "warranty"],
            "reviewer_challenge": "Is augmentation capex separately modelled and backed by warranty / supplier support?",
        },
        {
            "risk_id": "R04_OFFTAKER_CREDIT",
            "risk": "Offtaker credit and payment security",
            "asset_class": "Renewables",
            "terms": ["offtaker credit", "payment security", "letter of credit", "counterparty"],
            "reviewer_challenge": "Has offtaker creditworthiness and payment security been independently verified?",
        },
        {
            "risk_id": "R05_LAND_PERMITTING",
            "risk": "Land, easements and permits",
            "asset_class": "Renewables",
            "terms": ["land", "easement", "permit", "permitting"],
            "reviewer_challenge": "Are all land rights, easements and construction permits fully closed before approval?",
        },
        {
            "risk_id": "R06_CYBERSECURITY",
            "risk": "Cybersecurity / EMS / SCADA",
            "asset_class": "Solar / BESS",
            "terms": ["cybersecurity", "cyber", "ems", "scada"],
            "reviewer_challenge": "Has a dedicated EMS / SCADA cybersecurity assessment been completed?",
        },
        {
            "risk_id": "R07_DECOMMISSIONING",
            "risk": "Decommissioning and battery disposal",
            "asset_class": "Solar / BESS",
            "terms": ["decommissioning", "battery-disposal", "battery disposal"],
            "reviewer_challenge": "Is decommissioning and battery-disposal funding included in the model?",
        },
    ]

    benchmark_whitelist = [
        {
            "source_name": "02_Benchmark_and_Market_Data",
            "allowed_use": "Cost benchmarking only unless valuation-multiple evidence is specifically retrieved",
        },
        {
            "source_name": "NREL / ATB style benchmark workbook",
            "allowed_use": "PV, BESS, PV-plus-battery cost, grid and O&M benchmark categories",
        },
    ]

    return {
        "completeness_checklist": completeness_checklist,
        "strategy_criteria": strategy_criteria,
        "common_ic_questions": common_ic_questions,
        "asset_class_risk_taxonomy": asset_class_risk_taxonomy,
        "benchmark_whitelist": benchmark_whitelist,
    }


# ---------------------------------------------------------------------
# Retrieval plan
# ---------------------------------------------------------------------

def get_retrieval_plan(transaction_id):
    return {
        "profile": [
            {
                "query": (
                    f"{transaction_id} transaction overview asset geography technology ownership stage "
                    "solar BESS contracted revenue merchant revenue project cost equity IRR COD"
                ),
                "corpus_zone": "client_data",
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "completeness_readiness": [
            {
                "query": (
                    "required submission components completeness readiness missing weak open items "
                    "conditions precedent final investment approval metric reconciliation merchant price "
                    "offtaker credit combined downside BESS augmentation permit cyber decommissioning"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "strategy_fit": [
            {
                "query": (
                    "strategy fit growth priorities geography technology portfolio concentration "
                    "capital allocation control rights Thailand solar storage BESS contracted revenue"
                ),
                "corpus_zone": "client_data",
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "historical_ic_questions": [
            {
                "query": (
                    "historical IC questions recurring IC themes merchant exposure construction technology risk "
                    "counterparty credit downside protection risk allocation merchant price curves business case "
                    "seller assumptions IRR bridge accounting impact book values follow up actions"
                ),
                "corpus_zone": None,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "risk_review": [
            {
                "query": (
                    "principal risks mitigants risk rating merchant exposure grid cost schedule BESS degradation "
                    "offtaker credit land permits cybersecurity decommissioning curtailment construction technology risk"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "financial_reconciliation": [
            {
                "query": (
                    "main financial metrics returns project cost project IRR equity IRR NPV minimum DSCR "
                    "EV EBITDA debt total cost year one revenue metrics differ deck memo workbook"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "sensitivity_review": [
            {
                "query": (
                    "sensitivity results downside cases merchant price generation capex COD delay FX depreciation "
                    "combined downside BESS degradation augmentation equity IRR minimum DSCR included in submission"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "market_offtake_revenue": [
            {
                "query": (
                    "market offtake revenue assumptions contracted share merchant share PPA price "
                    "merchant price BESS revenue payment security offtaker credit merchant curve revenue projection EBITDA"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "client_valuation": [
            {
                "query": (
                    "valuation assumptions NPV EV EBITDA enterprise value first full year EBITDA "
                    "discount rate cost per MW BESS equipment grid interconnection overhead contingency benchmark"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "external_cost_benchmark": [
            {
                "query": (
                    "utility scale PV plus battery overnight capital cost grid connection cost "
                    "fixed operating expense variable operating expense battery storage solar PV benchmark"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "external_ev_ebitda_benchmark": [
            {
                "query": (
                    "renewable energy solar storage EV EBITDA enterprise value valuation multiple "
                    "transaction benchmark comparable transactions listed company multiples"
                ),
                "corpus_zone": "corpus_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 12000,
            }
        ],
        "conditions_open_items": [
            {
                "query": (
                    "conditions precedent open items final investment approval metric reconciliation "
                    "merchant price offtaker credit combined downside BESS augmentation grid overhead advisory "
                    "land permit cyber decommissioning actions"
                ),
                "corpus_zone": "client_data",
                "top_k": 20,
                "mode": "hybrid",
                "max_chunk_chars": 10000,
            }
        ],
        "previous_request_accommodation": [
            {
                "query": (
                    "previous request missing data accommodated prior IC request follow up action "
                    "previously requested included in current memo not accommodated"
                ),
                "corpus_zone": None,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "prior_deal_comparison": [
            {
                "query": (
                    "comparable prior investments historical deal comparison similarities differences "
                    "prior renewable investments platform solar storage wind BESS geography stage"
                ),
                "corpus_zone": None,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
        "macro_context": [
            {
                "query": (
                    "Thailand macroeconomic data FX projections exchange rate GDP growth inflation "
                    "renewable energy market country risk"
                ),
                "corpus_zone": None,
                "top_k": 15,
                "mode": "hybrid",
                "max_chunk_chars": 8000,
            }
        ],
    }


def run_single_retrieval(query_config):
    return ret.retrieve_chunks(
        query_text=query_config["query"],
        top_k=query_config.get("top_k", 10),
        mode=query_config.get("mode", "hybrid"),
        corpus_zone=query_config.get("corpus_zone"),
        corpus_pack=query_config.get("corpus_pack"),
        document_id=query_config.get("document_id"),
        max_chunk_chars=query_config.get("max_chunk_chars", 8000),
    )


def run_retrieval_plan(transaction_id):
    retrieval_plan = get_retrieval_plan(transaction_id)
    evidence_packets = {}

    for module_key, query_configs in retrieval_plan.items():
        module_results = []
        retrieval_calls = []

        for query_config in query_configs:
            try:
                retrieval_output = run_single_retrieval(query_config)
                results = retrieval_output.get("results", [])
                module_results.extend(results)

                retrieval_calls.append(
                    {
                        "query": query_config["query"],
                        "corpus_zone": query_config.get("corpus_zone"),
                        "results_returned": len(results),
                        "status": "ok",
                    }
                )

            except Exception as e:
                retrieval_calls.append(
                    {
                        "query": query_config["query"],
                        "corpus_zone": query_config.get("corpus_zone"),
                        "results_returned": 0,
                        "status": "failed",
                        "error": f"{type(e).__name__}: {str(e)}",
                    }
                )

        selected_results = select_evidence(
            results=module_results,
            module_key=module_key,
            top_n=12,
        )

        evidence_packets[module_key] = {
            "retrieval_calls": retrieval_calls,
            "raw_results_count": len(module_results),
            "selected_results": selected_results,
            "selected_sources": [get_source_reference(item) for item in selected_results],
            "evidence_blob": get_evidence_blob(selected_results),
        }

    return evidence_packets


# ---------------------------------------------------------------------
# Transaction profile
# ---------------------------------------------------------------------

def identify_transaction_profile(transaction_id, evidence_packets):
    profile_evidence = evidence_packets.get("profile", {}).get("selected_results", [])
    blob = get_evidence_blob(profile_evidence)

    technologies = []

    if has_any(blob, ["solar", "pv"]):
        technologies.append("Solar PV")

    if has_any(blob, ["bess", "battery", "storage"]):
        technologies.append("BESS")

    geography = "Not identified"
    if "thailand" in normalize_text(blob):
        geography = "Thailand"

    stage = "Not identified"
    if has_any(blob, ["late-stage development", "development"]):
        stage = "Development / late-stage development"

    ownership = "Not identified"
    if has_any(blob, ["80%", "80% controlling", "controlling interest", "controlling stake"]):
        ownership = "80% controlling stake"

    revenue_model = "Not identified"
    if has_any(blob, ["70% contracted", "contracted share", "merchant share", "30% merchant"]):
        revenue_model = "70% contracted / 30% merchant exposure"

    return {
        "transaction_id": transaction_id,
        "technologies": technologies,
        "geography": geography,
        "stage": stage,
        "ownership": ownership,
        "revenue_model": revenue_model,
        "source_chunks": [get_source_reference(item) for item in profile_evidence[:6]],
    }


# ---------------------------------------------------------------------
# SOW module builders
# ---------------------------------------------------------------------

def run_completeness_check(review_config, evidence_packets):
    checklist = review_config["completeness_checklist"]
    evidence_results = evidence_packets.get("completeness_readiness", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    rows = []

    for item in checklist:
        evidence_found = has_any(blob, item["evidence_terms"])
        weak_evidence = has_any(blob, item["weak_terms"])

        if not evidence_found:
            status = READINESS_MISSING
            traffic_light = STATUS_RED
            weakness = "Required evidence not found in retrieved submission materials."
        elif weak_evidence:
            status = READINESS_WEAK
            traffic_light = STATUS_AMBER
            weakness = "Evidence found, but weakness / open issue language is present."
        else:
            status = READINESS_PASS
            traffic_light = STATUS_GREEN
            weakness = "No obvious weakness language detected in retrieved evidence."

        evidence_text = short_evidence_text(
            results=evidence_results,
            terms=item["evidence_terms"] + item["weak_terms"],
            max_chars=700,
        )

        source_chunks = find_source_chunks(
            results=evidence_results,
            terms=item["evidence_terms"] + item["weak_terms"],
            max_sources=5,
        )

        if status == READINESS_PASS:
            recommended_action = "No immediate action identified from retrieved evidence."
        elif status == READINESS_WEAK:
            recommended_action = "Reviewer should request clarification / supporting evidence before final approval."
        else:
            recommended_action = "Reviewer should request missing item or confirm whether it is outside the current submission."

        rows.append(
            {
                "item_id": item["item_id"],
                "required_item": item["required_item"],
                "status": status,
                "traffic_light": traffic_light,
                "evidence_found": evidence_found,
                "weakness": weakness,
                "evidence_text": evidence_text,
                "source_chunks": source_chunks,
                "recommended_action": recommended_action,
            }
        )

    module_status = STATUS_GREEN

    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER

    return {
        "module": "Completeness & Readiness Check",
        "traffic_light": module_status,
        "summary": "Submission readiness assessed using deterministic checklist and retrieved evidence.",
        "checklist": rows,
    }


def run_strategy_fit_assessment(review_config, evidence_packets):
    criteria = review_config["strategy_criteria"]
    evidence_results = evidence_packets.get("strategy_fit", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    rows = []

    for criterion in criteria:
        if not criterion["configured"]:
            status = READINESS_NOT_ASSESSED
            traffic_light = STATUS_GREY
            finding = "Criterion is not configured for this POC run."
            source_chunks = []
        else:
            positive_found = has_any(blob, criterion["positive_terms"])
            negative_found = has_any(blob, criterion["negative_terms"])

            if positive_found and not negative_found:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                finding = "Alignment evidence found."
            elif positive_found and negative_found:
                status = READINESS_PARTIAL
                traffic_light = STATUS_AMBER
                finding = "Some alignment evidence found, but unresolved issue / limitation language is present."
            else:
                status = READINESS_NOT_ASSESSED
                traffic_light = STATUS_GREY
                finding = "Alignment could not be assessed from retrieved evidence."

            source_chunks = find_source_chunks(
                results=evidence_results,
                terms=criterion["positive_terms"] + criterion["negative_terms"],
                max_sources=5,
            )

        rows.append(
            {
                "criterion_id": criterion["criterion_id"],
                "criterion": criterion["criterion"],
                "configured": criterion["configured"],
                "status": status,
                "traffic_light": traffic_light,
                "finding": finding,
                "source_chunks": source_chunks,
            }
        )

    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER
    elif any(row["traffic_light"] == STATUS_GREY for row in rows):
        module_status = STATUS_AMBER
    else:
        module_status = STATUS_GREEN

    return {
        "module": "Strategy & Fit Assessment",
        "traffic_light": module_status,
        "summary": "Strategy fit is assessed using explicit configured rules. Unconfigured criteria are marked Grey.",
        "criteria_assessment": rows,
    }


def run_historical_ic_question_check(review_config, evidence_packets):
    questions = review_config["common_ic_questions"]
    evidence_results = evidence_packets.get("historical_ic_questions", {}).get("selected_results", [])

    current_submission_results = []
    for module_key in [
        "risk_review",
        "financial_reconciliation",
        "sensitivity_review",
        "market_offtake_revenue",
        "conditions_open_items",
    ]:
        current_submission_results.extend(evidence_packets.get(module_key, {}).get("selected_results", []))

    historical_blob = get_evidence_blob(evidence_results)
    current_blob = get_evidence_blob(current_submission_results)

    historical_logs_available = has_any(
        historical_blob,
        ["ic question", "investment committee", "follow-up action", "historical ic", "past ic"],
    )

    rows = []

    for question in questions:
        coverage_found = has_any(current_blob, question["coverage_terms"])
        weak_found = has_any(current_blob, question["weak_terms"])

        if coverage_found and not weak_found:
            coverage_status = "Fully addressed"
            traffic_light = STATUS_GREEN
        elif coverage_found and weak_found:
            coverage_status = "Partially addressed"
            traffic_light = STATUS_AMBER
        else:
            coverage_status = "Not addressed"
            traffic_light = STATUS_RED

        source_chunks = find_source_chunks(
            results=current_submission_results,
            terms=question["coverage_terms"] + question["weak_terms"],
            max_sources=5,
        )

        rows.append(
            {
                "question_id": question["question_id"],
                "theme": question["theme"],
                "likely_ic_question": question["question"],
                "coverage_status": coverage_status,
                "traffic_light": traffic_light,
                "source_chunks": source_chunks,
            }
        )

    return {
        "module": "Historical IC Question Coverage",
        "traffic_light": STATUS_AMBER,
        "historical_logs_available": historical_logs_available,
        "limitation": (
            "Historical IC Q&A logs / follow-up actions were not clearly evidenced in the current corpus. "
            "For this POC run, the tool uses a standard IC question library and does not claim to have learned "
            "from actual historical IC interactions."
        ),
        "question_coverage": rows,
    }


# ---------------------------------------------------------------------
# Financial reconciliation
# ---------------------------------------------------------------------

def infer_source_label(result):
    evidence_type = classify_evidence_type(result)

    if evidence_type == "client_workbook_control":
        return "workbook"
    if evidence_type == "client_memo":
        return "memo"
    if evidence_type == "client_deck":
        return "deck"
    if evidence_type == "client_pdf":
        return "pdf_deck"

    return "unknown"


def extract_value_near_metric(text_value, metric_terms):
    text_value = clean_text(text_value)

    for term in metric_terms:
        pattern = rf"{re.escape(term)}[^0-9\-]{{0,80}}([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, text_value, flags=re.IGNORECASE)
        if match:
            return safe_float(match.group(1))

    row_lines = re.findall(r"Row\s+\d+:[^R]+", text_value)

    for row in row_lines:
        row_lower = row.lower()

        if not any(term.lower() in row_lower for term in metric_terms):
            continue

        match = re.search(r"\|\s*1:\s*([0-9]+(?:\.[0-9]+)?)", row)
        if match:
            return safe_float(match.group(1))

        match = re.search(r"Unnamed:\s*2:\s*([0-9]+(?:\.[0-9]+)?)", row, flags=re.IGNORECASE)
        if match:
            return safe_float(match.group(1))

        match = re.search(r"Unnamed:\s*1:\s*([0-9]+(?:\.[0-9]+)?)", row, flags=re.IGNORECASE)
        if match:
            return safe_float(match.group(1))

    return None


def extract_financial_metrics(evidence_results):
    metric_definitions = [
        {"metric_key": "total_project_cost_usd_mn", "metric_name": "Total project cost", "terms": ["Total project cost", "project cost"], "unit": "USD mn"},
        {"metric_key": "project_irr_pct", "metric_name": "Project IRR", "terms": ["Project IRR"], "unit": "%"},
        {"metric_key": "equity_irr_pct", "metric_name": "Equity IRR", "terms": ["Equity IRR"], "unit": "%"},
        {"metric_key": "npv_usd_mn", "metric_name": "NPV", "terms": ["NPV"], "unit": "USD mn"},
        {"metric_key": "minimum_dscr_x", "metric_name": "Minimum DSCR", "terms": ["Minimum DSCR", "Min_DSCR"], "unit": "x"},
        {"metric_key": "ev_ebitda_x", "metric_name": "EV / EBITDA", "terms": ["EV / EBITDA", "EV EBITDA"], "unit": "x"},
        {"metric_key": "debt_total_cost_pct", "metric_name": "Debt / total cost", "terms": ["Debt / total cost", "debt total cost"], "unit": "%"},
        {"metric_key": "year_one_revenue_usd_mn", "metric_name": "Year-one revenue", "terms": ["Year-one revenue", "year one revenue"], "unit": "USD mn"},
        {"metric_key": "ppa_price_usd_mwh", "metric_name": "PPA price", "terms": ["PPA price", "Deck PPA price"], "unit": "USD/MWh"},
        {"metric_key": "capacity_factor_pct", "metric_name": "Capacity factor", "terms": ["Capacity factor"], "unit": "%"},
    ]

    rows = []

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        source_label = infer_source_label(result)

        for metric in metric_definitions:
            if not has_any(chunk_text, metric["terms"]):
                continue

            value = extract_value_near_metric(chunk_text, metric["terms"])

            if value is None:
                continue

            rows.append(
                {
                    "metric_key": metric["metric_key"],
                    "metric_name": metric["metric_name"],
                    "source_label": source_label,
                    "value": value,
                    "unit": metric["unit"],
                    "source_chunk": get_source_reference(result),
                }
            )

    return rows


def reconcile_financial_metrics(metric_rows):
    by_metric = {}

    for row in metric_rows:
        metric_key = row["metric_key"]

        if metric_key not in by_metric:
            by_metric[metric_key] = {
                "metric_key": metric_key,
                "metric_name": row["metric_name"],
                "unit": row["unit"],
                "values_by_source": {},
                "source_chunks": [],
            }

        source_label = row["source_label"]
        value = row["value"]

        if source_label not in by_metric[metric_key]["values_by_source"]:
            by_metric[metric_key]["values_by_source"][source_label] = value

        by_metric[metric_key]["source_chunks"].append(row["source_chunk"])

    reconciliation_rows = []

    for metric_key, item in by_metric.items():
        values = list(item["values_by_source"].values())
        rounded_values = sorted(set([round(float(value), 4) for value in values if value is not None]))

        if len(rounded_values) > 1:
            traffic_light = STATUS_RED
            issue = "Inconsistent values found across sources."
        elif len(rounded_values) == 1:
            traffic_light = STATUS_GREEN
            issue = "No inconsistency detected across retrieved values."
        else:
            traffic_light = STATUS_GREY
            issue = "No numeric value extracted."

        preferred_value = None

        if "workbook" in item["values_by_source"]:
            preferred_value = item["values_by_source"]["workbook"]
        elif "memo" in item["values_by_source"]:
            preferred_value = item["values_by_source"]["memo"]
        elif "deck" in item["values_by_source"]:
            preferred_value = item["values_by_source"]["deck"]
        elif values:
            preferred_value = values[0]

        reconciliation_rows.append(
            {
                "metric_key": metric_key,
                "metric_name": item["metric_name"],
                "unit": item["unit"],
                "values_by_source": item["values_by_source"],
                "preferred_value_for_review": preferred_value,
                "traffic_light": traffic_light,
                "issue": issue,
                "source_chunks": item["source_chunks"],
            }
        )

    return reconciliation_rows


def run_financial_reconciliation(evidence_packets):
    evidence_results = evidence_packets.get("financial_reconciliation", {}).get("selected_results", [])
    metric_rows = extract_financial_metrics(evidence_results)
    reconciliation_rows = reconcile_financial_metrics(metric_rows)

    if any(row["traffic_light"] == STATUS_RED for row in reconciliation_rows):
        module_status = STATUS_RED
    elif reconciliation_rows:
        module_status = STATUS_GREEN
    else:
        module_status = STATUS_GREY

    return {
        "module": "Financial Metrics and Inconsistency Review",
        "traffic_light": module_status,
        "summary": "Financial metrics are extracted deterministically and reconciled across retrieved sources.",
        "metric_rows_extracted": metric_rows,
        "reconciliation_table": reconciliation_rows,
    }


# ---------------------------------------------------------------------
# Sensitivity review
# ---------------------------------------------------------------------

def extract_sensitivity_cases(evidence_results):
    sensitivity_cases = []

    scenario_terms = [
        "Base Case",
        "Merchant price -15%",
        "Generation -5%",
        "Capex +10%",
        "COD delay 6 months",
        "FX depreciation 10%",
        "Combined downside",
        "BESS degradation / augmentation",
    ]

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        rows = re.findall(r"Row\s+\d+:[^R]+", chunk_text)

        for row in rows:
            if not has_any(row, scenario_terms):
                continue

            scenario = None
            for term in scenario_terms:
                if has_any(row, [term]):
                    scenario = term
                    break

            equity_irr = None
            min_dscr = None
            included_in_submission = None

            match = re.search(r"Unnamed:\s*6:\s*([0-9]+(?:\.[0-9]+)?)", row, flags=re.IGNORECASE)
            if match:
                equity_irr = safe_float(match.group(1))

            match = re.search(r"Unnamed:\s*7:\s*([0-9]+(?:\.[0-9]+)?)", row, flags=re.IGNORECASE)
            if match:
                min_dscr = safe_float(match.group(1))

            match = re.search(r"Unnamed:\s*8:\s*(Yes|No)", row, flags=re.IGNORECASE)
            if match:
                included_in_submission = match.group(1)

            if equity_irr is None:
                match = re.search(r"\|\s*1:\s*([0-9]+(?:\.[0-9]+)?)%?", row)
                if match:
                    equity_irr = safe_float(match.group(1))

            if min_dscr is None:
                match = re.search(r"\|\s*2:\s*([0-9]+(?:\.[0-9]+)?)x?", row)
                if match:
                    min_dscr = safe_float(match.group(1))

            sensitivity_cases.append(
                {
                    "scenario": scenario,
                    "equity_irr_pct": equity_irr,
                    "minimum_dscr_x": min_dscr,
                    "included_in_submission": included_in_submission,
                    "source_chunk": get_source_reference(result),
                }
            )

    return sensitivity_cases


def run_sensitivity_review(evidence_packets):
    evidence_results = evidence_packets.get("sensitivity_review", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)
    sensitivity_cases = extract_sensitivity_cases(evidence_results)

    combined_downside_available = has_any(blob, ["combined downside"])
    bess_augmentation_available = has_any(blob, ["bess degradation", "augmentation", "battery augmentation"])

    combined_downside_not_in_submission = has_any(
        blob,
        [
            "combined downside not shown",
            "combined downside and bess augmentation sensitivities are absent",
            "sensitivity cases: combined downside",
            "unnamed: 8: no",
        ],
    )

    if not sensitivity_cases and not combined_downside_available:
        module_status = STATUS_RED
    elif combined_downside_not_in_submission:
        module_status = STATUS_AMBER
    else:
        module_status = STATUS_GREEN

    return {
        "module": "Sensitivity and Downside Protection Review",
        "traffic_light": module_status,
        "summary": "Sensitivity review distinguishes submitted cases from workbook-only cases.",
        "combined_downside_available_in_evidence": combined_downside_available,
        "bess_augmentation_available_in_evidence": bess_augmentation_available,
        "submission_gap_flag": combined_downside_not_in_submission,
        "sensitivity_cases": sensitivity_cases,
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


# ---------------------------------------------------------------------
# Market/offtake/revenue
# ---------------------------------------------------------------------

def run_market_offtake_revenue_review(evidence_packets):
    evidence_results = evidence_packets.get("market_offtake_revenue", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    flags = {
        "contracted_merchant_split_evidenced": has_any(blob, ["contracted share", "70%", "merchant share", "30%"]),
        "merchant_curve_missing": has_any(blob, ["merchant curve source not appended", "independent merchant curve referenced but not appended", "independent merchant-price study is referenced but not included"]),
        "offtaker_credit_missing": has_any(blob, ["no independent credit", "offtaker-credit support are missing", "offtaker credit support are missing"]),
        "payment_security_pending": has_any(blob, ["payment-security package remains under negotiation", "payment security", "under discussion", "not executed"]),
        "ppa_price_conflict": has_any(blob, ["deck says 60", "deck ppa price", "ppa price 2029"]),
        "bess_revenue_evidenced": has_any(blob, ["bess revenue", "ancillary services", "capacity", "shifting"]),
    }

    findings = []

    if flags["contracted_merchant_split_evidenced"]:
        findings.append(
            {
                "finding": "Revenue model includes contracted and merchant components.",
                "traffic_light": STATUS_AMBER if flags["merchant_curve_missing"] else STATUS_GREEN,
                "explanation": "Contracted / merchant split is evidenced, but merchant support must be checked.",
                "source_chunks": find_source_chunks(evidence_results, ["contracted share", "merchant share", "70%", "30%"]),
            }
        )

    if flags["merchant_curve_missing"]:
        findings.append(
            {
                "finding": "Independent merchant-price curve support is missing or not appended.",
                "traffic_light": STATUS_RED,
                "explanation": "The submission appears to rely on merchant revenue, but the independent curve is not available in the submission evidence.",
                "source_chunks": find_source_chunks(evidence_results, ["merchant curve", "not appended", "not included"]),
            }
        )

    if flags["offtaker_credit_missing"] or flags["payment_security_pending"]:
        findings.append(
            {
                "finding": "Offtaker credit and payment-security package are not fully evidenced.",
                "traffic_light": STATUS_RED,
                "explanation": "Payment security / credit support appears under discussion, not fully executed or independently reviewed.",
                "source_chunks": find_source_chunks(evidence_results, ["offtaker", "credit", "payment security", "under negotiation", "under discussion"]),
            }
        )

    if flags["ppa_price_conflict"]:
        findings.append(
            {
                "finding": "PPA price inconsistency should be reconciled.",
                "traffic_light": STATUS_AMBER,
                "explanation": "Retrieved evidence indicates different PPA price references across sources.",
                "source_chunks": find_source_chunks(evidence_results, ["PPA price", "deck says 60", "58", "60"]),
            }
        )

    module_status = STATUS_GREEN
    if any(item["traffic_light"] == STATUS_RED for item in findings):
        module_status = STATUS_RED
    elif any(item["traffic_light"] == STATUS_AMBER for item in findings):
        module_status = STATUS_AMBER

    return {
        "module": "Market, Offtake and Revenue Review",
        "traffic_light": module_status,
        "flags": flags,
        "findings": findings,
    }


# ---------------------------------------------------------------------
# Risk register
# ---------------------------------------------------------------------

def extract_rating_for_risk(blob, terms):
    blob_lower = normalize_text(blob)

    for term in terms:
        term_lower = normalize_text(term)
        position = blob_lower.find(term_lower)

        if position == -1:
            continue

        window = blob_lower[position:position + 300]

        if "not assessed" in window:
            return "Not assessed"
        if "medium-high" in window:
            return "Medium-high"
        if "high" in window:
            return "High"
        if "medium" in window:
            return "Medium"
        if "low" in window:
            return "Low"

    return "Not specified"


def run_risk_review(review_config, evidence_packets):
    risk_taxonomy = review_config["asset_class_risk_taxonomy"]
    evidence_results = evidence_packets.get("risk_review", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    rows = []

    for risk_item in risk_taxonomy:
        risk_found = has_any(blob, risk_item["terms"])

        if not risk_found:
            rows.append(
                {
                    "risk_id": risk_item["risk_id"],
                    "risk": risk_item["risk"],
                    "asset_class": risk_item["asset_class"],
                    "rating": "Not identified",
                    "mitigation_status": "No evidence found in retrieved materials.",
                    "reviewer_challenge": risk_item["reviewer_challenge"],
                    "open_gap": "Risk topic not clearly addressed in retrieved evidence.",
                    "traffic_light": STATUS_GREY,
                    "source_chunks": [],
                }
            )
            continue

        rating = extract_rating_for_risk(blob, risk_item["terms"])
        source_chunks = find_source_chunks(evidence_results, risk_item["terms"], max_sources=5)
        risk_window_text = short_evidence_text(evidence_results, risk_item["terms"], max_chars=800)

        weak = has_any(
            risk_window_text,
            ["not assessed", "not modelled", "not appended", "under negotiation", "absent", "open", "missing"],
        )

        if rating == "Not assessed":
            traffic_light = STATUS_RED
        elif rating in ["High", "Medium-high"] or weak:
            traffic_light = STATUS_AMBER
        else:
            traffic_light = STATUS_GREEN

        rows.append(
            {
                "risk_id": risk_item["risk_id"],
                "risk": risk_item["risk"],
                "asset_class": risk_item["asset_class"],
                "rating": rating,
                "mitigation_status": risk_window_text,
                "reviewer_challenge": risk_item["reviewer_challenge"],
                "open_gap": "Review the mitigation/status evidence and request support where weak language is present.",
                "traffic_light": traffic_light,
                "source_chunks": source_chunks,
            }
        )

    module_status = STATUS_GREEN
    if any(row["traffic_light"] == STATUS_RED for row in rows):
        module_status = STATUS_RED
    elif any(row["traffic_light"] == STATUS_AMBER for row in rows):
        module_status = STATUS_AMBER
    elif any(row["traffic_light"] == STATUS_GREY for row in rows):
        module_status = STATUS_AMBER

    return {
        "module": "Risk Register and Reviewer Challenge Prompts",
        "traffic_light": module_status,
        "risk_register": rows,
    }


# ---------------------------------------------------------------------
# Valuation and benchmark review
# ---------------------------------------------------------------------

def run_client_valuation_review(evidence_packets):
    evidence_results = evidence_packets.get("client_valuation", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    valuation_metrics_present = has_any(blob, ["npv", "ev / ebitda", "ev ebitda", "minimum dscr", "equity irr"])
    cost_challenge_present = has_any(blob, ["bess equipment", "grid interconnection", "overhead", "contingency", "advisory", "upper quartile", "high"])

    findings = []

    if valuation_metrics_present:
        findings.append(
            {
                "finding": "Client-side valuation metrics are evidenced.",
                "traffic_light": STATUS_GREEN,
                "explanation": "Retrieved evidence includes valuation/return metrics such as NPV, DSCR and EV/EBITDA.",
                "source_chunks": find_source_chunks(evidence_results, ["NPV", "EV / EBITDA", "minimum DSCR", "equity IRR"]),
            }
        )
    else:
        findings.append(
            {
                "finding": "Client-side valuation metrics not clearly evidenced.",
                "traffic_light": STATUS_RED,
                "explanation": "The retrieval did not find sufficient client-side valuation metric evidence.",
                "source_chunks": [],
            }
        )

    if cost_challenge_present:
        findings.append(
            {
                "finding": "Project-cost challenge areas are evidenced.",
                "traffic_light": STATUS_AMBER,
                "explanation": (
                    "Evidence points to BESS equipment, grid interconnection, contingency, overhead "
                    "or advisory items requiring challenge. These are client-side assertions unless "
                    "independently benchmarked through corpus_data."
                ),
                "source_chunks": find_source_chunks(
                    evidence_results,
                    ["BESS equipment", "grid interconnection", "overhead", "contingency", "advisory", "upper quartile", "high"],
                ),
            }
        )

    return {
        "module": "Client-Side Valuation Review",
        "traffic_light": STATUS_AMBER if cost_challenge_present else STATUS_GREEN,
        "important_note": "Client-side benchmark statements are not treated as independent external benchmark confirmation.",
        "findings": findings,
    }


def run_external_benchmark_review(evidence_packets):
    cost_results = evidence_packets.get("external_cost_benchmark", {}).get("selected_results", [])
    ev_results = evidence_packets.get("external_ev_ebitda_benchmark", {}).get("selected_results", [])

    cost_blob = get_evidence_blob(cost_results)
    ev_blob = get_evidence_blob(ev_results)

    cost_benchmark_available = has_any(
        cost_blob,
        [
            "utility-scale battery storage",
            "utility-scale pv",
            "pv-plus-battery",
            "overnight capital cost",
            "grid connection cost",
            "fixed operating expenses",
        ],
    )

    ev_ebitda_benchmark_available = has_all(ev_blob, ["ev", "ebitda"]) and has_any(
        ev_blob,
        ["enterprise value", "valuation multiple", "transaction benchmark", "comparable transaction", "listed company"],
    )

    findings = []

    if cost_benchmark_available:
        findings.append(
            {
                "benchmark_area": "External cost benchmark",
                "traffic_light": STATUS_GREEN,
                "finding": "External cost benchmark categories are available for PV / BESS / PV-plus-battery.",
                "allowed_use": "Use for cost, grid and O&M benchmarking only.",
                "source_chunks": find_source_chunks(
                    cost_results,
                    ["utility-scale battery storage", "utility-scale pv", "pv-plus-battery", "overnight capital cost", "grid connection cost"],
                ),
            }
        )
    else:
        findings.append(
            {
                "benchmark_area": "External cost benchmark",
                "traffic_light": STATUS_GREY,
                "finding": "External cost benchmark evidence was not found.",
                "allowed_use": "Do not make external cost benchmark claims.",
                "source_chunks": [],
            }
        )

    if ev_ebitda_benchmark_available:
        findings.append(
            {
                "benchmark_area": "External EV/EBITDA benchmark",
                "traffic_light": STATUS_GREEN,
                "finding": "External EV/EBITDA valuation multiple evidence appears available.",
                "allowed_use": "May be used cautiously if sources are verified.",
                "source_chunks": find_source_chunks(ev_results, ["EV", "EBITDA", "enterprise value", "valuation multiple"]),
            }
        )
    else:
        findings.append(
            {
                "benchmark_area": "External EV/EBITDA benchmark",
                "traffic_light": STATUS_GREY,
                "finding": "External valuation multiple benchmark evidence was not found in the current corpus.",
                "allowed_use": "Do not claim EV/EBITDA is externally benchmarked.",
                "source_chunks": [],
            }
        )

    module_status = STATUS_AMBER
    if cost_benchmark_available and ev_ebitda_benchmark_available:
        module_status = STATUS_GREEN

    return {
        "module": "External Benchmark Review",
        "traffic_light": module_status,
        "cost_benchmark_available": cost_benchmark_available,
        "ev_ebitda_benchmark_available": ev_ebitda_benchmark_available,
        "findings": findings,
    }


# ---------------------------------------------------------------------
# Conditions and grey modules
# ---------------------------------------------------------------------

def build_open_items_register(evidence_packets):
    evidence_results = evidence_packets.get("conditions_open_items", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    open_item_definitions = [
        {"category": "Financial", "condition": "Reconcile financial metrics across deck, memo and workbook.", "terms": ["metric reconciliation", "reconcile financial metrics", "metrics differ", "reconciliation"], "priority": "High"},
        {"category": "Market/offtake", "condition": "Obtain independent merchant-price curve support.", "terms": ["merchant-price", "merchant price", "merchant curve", "not appended"], "priority": "High"},
        {"category": "Market/offtake", "condition": "Obtain offtaker-credit evidence and executed payment-security package.", "terms": ["offtaker credit", "payment security", "letter of credit", "under negotiation", "under discussion"], "priority": "High"},
        {"category": "Sensitivity", "condition": "Run / present combined downside and explicit BESS augmentation sensitivity.", "terms": ["combined downside", "bess augmentation", "battery augmentation", "sensitivity"], "priority": "High"},
        {"category": "Cost control", "condition": "Cap grid, overhead and advisory costs; review overlapping scopes.", "terms": ["grid", "overhead", "advisory", "cap", "overlapping"], "priority": "Medium-high"},
        {"category": "Legal/permitting", "condition": "Close land easements and permit updates.", "terms": ["land", "easement", "permit", "permitting"], "priority": "Medium-high"},
        {"category": "Cyber", "condition": "Complete cybersecurity / EMS / SCADA assessment.", "terms": ["cyber", "cybersecurity", "ems", "scada"], "priority": "Medium-high"},
        {"category": "Decommissioning", "condition": "Establish decommissioning / battery-disposal reserve or funding plan.", "terms": ["decommissioning", "battery-disposal", "battery disposal"], "priority": "Medium-high"},
    ]

    rows = []

    for item in open_item_definitions:
        found = has_any(blob, item["terms"])

        if found:
            traffic_light = STATUS_AMBER
            status = "Open / requires closure"
            evidence_text = short_evidence_text(evidence_results, item["terms"], max_chars=700)
            source_chunks = find_source_chunks(evidence_results, item["terms"], max_sources=5)
        else:
            traffic_light = STATUS_GREY
            status = "Not evidenced"
            evidence_text = ""
            source_chunks = []

        rows.append(
            {
                "category": item["category"],
                "condition": item["condition"],
                "priority": item["priority"],
                "status": status,
                "traffic_light": traffic_light,
                "evidence_text": evidence_text,
                "source_chunks": source_chunks,
                "recommended_action": "Track to closure before final approval." if found else "Confirm whether this condition is applicable.",
            }
        )

    return {
        "module": "Conditions Precedent and Open Items",
        "traffic_light": STATUS_AMBER,
        "open_items": rows,
    }


def run_previous_request_accommodation_check(evidence_packets):
    evidence_results = evidence_packets.get("previous_request_accommodation", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    available = has_any(blob, ["previous request", "prior request", "accommodated", "follow-up action"])

    if not available:
        return {
            "module": "Previous Request Accommodation",
            "traffic_light": STATUS_GREY,
            "status": READINESS_NOT_ASSESSED,
            "summary": (
                "Prior request / follow-up action evidence was not found in the current corpus. "
                "The report does not assess whether previous requests were accommodated."
            ),
            "required_future_data": "Load previous_ic_requests or follow_up_actions table / corpus.",
            "findings": [],
        }

    return {
        "module": "Previous Request Accommodation",
        "traffic_light": STATUS_AMBER,
        "status": READINESS_PARTIAL,
        "summary": "Potential prior request evidence found. Manual validation recommended.",
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


def run_prior_deal_comparison(evidence_packets):
    evidence_results = evidence_packets.get("prior_deal_comparison", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    available = has_any(blob, ["comparable prior", "historical deal", "previous transaction", "similarities", "differences"])

    if not available:
        return {
            "module": "Historical Deal Comparison",
            "traffic_light": STATUS_GREY,
            "status": READINESS_NOT_ASSESSED,
            "summary": (
                "Comparable prior-deal evidence was not found in the current corpus. "
                "The report does not claim automatic historical deal comparison."
            ),
            "required_future_data": "Load deal_metadata / prior IC materials with asset class, geography, size, stage, ownership and revenue model.",
            "comparables": [],
        }

    return {
        "module": "Historical Deal Comparison",
        "traffic_light": STATUS_AMBER,
        "status": READINESS_PARTIAL,
        "summary": "Potential comparable-deal evidence found. Manual validation recommended.",
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


def run_macro_context_review(evidence_packets):
    evidence_results = evidence_packets.get("macro_context", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    available = has_any(blob, ["fx", "exchange rate", "gdp", "inflation", "macroeconomic", "country risk"])

    if not available:
        return {
            "module": "Macro / FX / GDP Context",
            "traffic_light": STATUS_GREY,
            "status": READINESS_NOT_ASSESSED,
            "summary": (
                "Macro / FX / GDP evidence was not found in the current corpus. "
                "The report does not fabricate macroeconomic conclusions."
            ),
            "required_future_data": "Connect or load whitelisted macro / FX / GDP source data.",
            "findings": [],
        }

    return {
        "module": "Macro / FX / GDP Context",
        "traffic_light": STATUS_AMBER,
        "status": READINESS_PARTIAL,
        "summary": "Potential macro evidence found. Manual validation recommended.",
        "source_chunks": [get_source_reference(item) for item in evidence_results[:8]],
    }


# ---------------------------------------------------------------------
# Executive summary and dashboard
# ---------------------------------------------------------------------

def build_traffic_light_dashboard(report_sections):
    dashboard = []

    for section_key, section_value in report_sections.items():
        if not isinstance(section_value, dict):
            continue

        dashboard.append(
            {
                "section_key": section_key,
                "section_name": section_value.get("module", section_key),
                "traffic_light": section_value.get("traffic_light", STATUS_GREY),
                "summary": section_value.get("summary", ""),
            }
        )

    return dashboard


def build_executive_summary(report_sections):
    dashboard = build_traffic_light_dashboard(report_sections)

    red_items = [item for item in dashboard if item["traffic_light"] == STATUS_RED]
    amber_items = [item for item in dashboard if item["traffic_light"] == STATUS_AMBER]
    grey_items = [item for item in dashboard if item["traffic_light"] == STATUS_GREY]

    if red_items:
        overall_status = STATUS_RED
    elif amber_items:
        overall_status = STATUS_AMBER
    elif grey_items:
        overall_status = STATUS_AMBER
    else:
        overall_status = STATUS_GREEN

    top_findings = []

    for item in red_items[:5]:
        top_findings.append({"traffic_light": STATUS_RED, "finding": f"{item['section_name']} requires immediate reviewer attention."})

    for item in amber_items[:5]:
        top_findings.append({"traffic_light": STATUS_AMBER, "finding": f"{item['section_name']} is partially addressed or contains unresolved issues."})

    for item in grey_items[:5]:
        top_findings.append({"traffic_light": STATUS_GREY, "finding": f"{item['section_name']} could not be fully assessed due to unavailable source/config data."})

    return {
        "module": "Executive IC Review Summary",
        "overall_readiness_status": overall_status,
        "summary": (
            "This output is an AI-enabled first-line IC review pack. It supports reviewer judgment "
            "and does not replace Investment Committee decision-making."
        ),
        "top_findings": top_findings,
        "red_count": len(red_items),
        "amber_count": len(amber_items),
        "grey_count": len(grey_items),
        "green_count": len([item for item in dashboard if item["traffic_light"] == STATUS_GREEN]),
    }


# ---------------------------------------------------------------------
# Evidence appendix
# ---------------------------------------------------------------------

def build_evidence_appendix(evidence_packets):
    source_map = {}

    for module_key, packet in evidence_packets.items():
        for result in packet.get("selected_results", []):
            source = get_source_reference(result)
            chunk_id = source.get("chunk_id")

            if not chunk_id:
                continue

            if chunk_id not in source_map:
                source_map[chunk_id] = source
                source_map[chunk_id]["used_in_modules"] = []

            source_map[chunk_id]["used_in_modules"].append(module_key)

    appendix = list(source_map.values())
    appendix = sorted(appendix, key=lambda x: str(x.get("chunk_id")))

    return appendix


# ---------------------------------------------------------------------
# Optional LLM summary
# ---------------------------------------------------------------------

def generate_llm_executive_summary(config_base, report_core):
    openai_api_key = config_base.get("openai_api_key")
    llm_model = config_base.get("llm_model")

    if not openai_api_key:
        return {
            "status": "not_generated",
            "reason": "OPENAI_API_KEY is not set.",
            "summary_text": "",
        }

    if not llm_model:
        return {
            "status": "not_generated",
            "reason": "llm_model is not set in config.",
            "summary_text": "",
        }

    prompt = f"""
You are preparing an AI first-line Investment Committee review summary.

Rules:
- Do not write a market research report.
- Do not make an investment recommendation.
- Do not invent facts.
- Use only the structured findings provided.
- Clearly flag Grey areas as not assessed due to missing source/config data.
- Focus on reviewer concerns, gaps, inconsistencies and follow-up questions.
- Keep the summary concise and IC-review oriented.

Structured findings:
{json.dumps(report_core, ensure_ascii=False, indent=2)[:25000]}

Generate:
1. One short executive paragraph.
2. Top red/amber issues.
3. Key IC follow-up questions.
4. Data limitations.
"""

    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.responses.create(
            model=llm_model,
            input=prompt,
        )

        summary_text = getattr(response, "output_text", "")

        return {
            "status": "generated",
            "model": llm_model,
            "summary_text": summary_text,
        }

    except Exception as responses_error:
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You generate concise IC first-line review summaries from structured evidence.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            summary_text = response.choices[0].message.content

            return {
                "status": "generated",
                "model": llm_model,
                "summary_text": summary_text,
            }

        except Exception as chat_error:
            return {
                "status": "not_generated",
                "reason": (
                    f"Responses API error: {type(responses_error).__name__}: {str(responses_error)[:500]} | "
                    f"Chat Completions fallback error: {type(chat_error).__name__}: {str(chat_error)[:500]}"
                ),
                "summary_text": "",
            }


# ---------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------

def traffic_icon(status):
    if status == STATUS_GREEN:
        return "🟢"
    if status == STATUS_AMBER:
        return "🟠"
    if status == STATUS_RED:
        return "🔴"
    if status == STATUS_GREY:
        return "⚪"
    return "⚪"


def format_sources_md(source_chunks):
    if not source_chunks:
        return "No source reference available."

    parts = []

    for src in source_chunks[:5]:
        parts.append(
            f"{src.get('chunk_id')} | {src.get('document_id')} | "
            f"{src.get('section_heading')} | {src.get('source_reference')}"
        )

    return "<br>".join(parts)


def format_markdown_report(report):
    lines = []

    lines.append("# AI First-Line IC Review Pack")
    lines.append("")
    lines.append(f"**Classification:** {report.get('classification')}")
    lines.append(f"**Transaction ID:** {report.get('transaction_id')}")
    lines.append(f"**Run ID:** {report.get('run_id')}")
    lines.append(f"**Generated at:** {report.get('generated_at')}")
    lines.append("")

    executive_summary = report.get("executive_summary", {})
    lines.append("## 1. Executive IC Review Summary")
    lines.append("")
    lines.append(f"**Overall status:** {traffic_icon(executive_summary.get('overall_readiness_status'))} {executive_summary.get('overall_readiness_status')}")
    lines.append("")
    lines.append(executive_summary.get("summary", ""))
    lines.append("")

    llm_summary = report.get("llm_executive_summary", {})
    if llm_summary.get("summary_text"):
        lines.append("### LLM-generated executive summary")
        lines.append("")
        lines.append(llm_summary.get("summary_text"))
        lines.append("")

    lines.append("### Top findings")
    lines.append("")
    for finding in executive_summary.get("top_findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} {finding.get('finding')}")
    lines.append("")

    lines.append("## 2. Traffic-Light Dashboard")
    lines.append("")
    lines.append("| Section | Status | Summary |")
    lines.append("|---|---|---|")
    for item in report.get("traffic_light_dashboard", []):
        lines.append(
            f"| {item.get('section_name')} | {traffic_icon(item.get('traffic_light'))} {item.get('traffic_light')} | {clean_text(item.get('summary'))} |"
        )
    lines.append("")

    completeness = report.get("completeness_readiness", {})
    lines.append("## 3. Completeness & Readiness Check")
    lines.append("")
    lines.append("| Required item | Status | Weakness | Recommended action | Sources |")
    lines.append("|---|---|---|---|---|")
    for row in completeness.get("checklist", []):
        lines.append(
            f"| {row.get('required_item')} | {traffic_icon(row.get('traffic_light'))} {row.get('status')} | "
            f"{clean_text(row.get('weakness'))} | {clean_text(row.get('recommended_action'))} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    strategy = report.get("strategy_fit_assessment", {})
    lines.append("## 4. Strategy & Fit Assessment")
    lines.append("")
    lines.append("| Criterion | Status | Finding | Sources |")
    lines.append("|---|---|---|---|")
    for row in strategy.get("criteria_assessment", []):
        lines.append(
            f"| {row.get('criterion')} | {traffic_icon(row.get('traffic_light'))} {row.get('status')} | "
            f"{clean_text(row.get('finding'))} | {format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    icq = report.get("historical_ic_question_coverage", {})
    lines.append("## 5. Historical IC Question Coverage")
    lines.append("")
    lines.append(f"**Limitation:** {icq.get('limitation', '')}")
    lines.append("")
    lines.append("| Theme | Likely IC question | Coverage | Sources |")
    lines.append("|---|---|---|---|")
    for row in icq.get("question_coverage", []):
        lines.append(
            f"| {row.get('theme')} | {row.get('likely_ic_question')} | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('coverage_status')} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    fin = report.get("financial_reconciliation", {})
    lines.append("## 6. Key Inconsistencies & Financial Metric Reconciliation")
    lines.append("")
    lines.append("| Metric | Values by source | Preferred review value | Status | Issue |")
    lines.append("|---|---|---:|---|---|")
    for row in fin.get("reconciliation_table", []):
        lines.append(
            f"| {row.get('metric_name')} | {json.dumps(row.get('values_by_source'), ensure_ascii=False)} | "
            f"{row.get('preferred_value_for_review')} {row.get('unit')} | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('traffic_light')} | {row.get('issue')} |"
        )
    lines.append("")

    market = report.get("market_offtake_revenue_review", {})
    lines.append("## 7. Market, Offtake and Revenue Review")
    lines.append("")
    for finding in market.get("findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} **{finding.get('finding')}** — {finding.get('explanation')}")
    lines.append("")

    sens = report.get("sensitivity_review", {})
    lines.append("## 8. Sensitivity and Downside Protection Review")
    lines.append("")
    lines.append(f"**Submission gap flag:** {sens.get('submission_gap_flag')}")
    lines.append("")
    lines.append("| Scenario | Equity IRR | Minimum DSCR | Included in submission | Source |")
    lines.append("|---|---:|---:|---|---|")
    for row in sens.get("sensitivity_cases", []):
        src = row.get("source_chunk", {})
        lines.append(
            f"| {row.get('scenario')} | {row.get('equity_irr_pct')} | {row.get('minimum_dscr_x')} | "
            f"{row.get('included_in_submission')} | {src.get('chunk_id')} |"
        )
    lines.append("")

    risk = report.get("risk_review", {})
    lines.append("## 9. Risk Register and Reviewer Prompts")
    lines.append("")
    lines.append("| Risk | Rating | Reviewer challenge | Status | Sources |")
    lines.append("|---|---|---|---|---|")
    for row in risk.get("risk_register", []):
        lines.append(
            f"| {row.get('risk')} | {row.get('rating')} | {row.get('reviewer_challenge')} | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('traffic_light')} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    valuation = report.get("valuation_review", {})
    external = report.get("external_benchmark_review", {})
    lines.append("## 10. Valuation and Benchmark Review")
    lines.append("")
    lines.append("### Client-side valuation")
    for finding in valuation.get("findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} **{finding.get('finding')}** — {finding.get('explanation')}")
    lines.append("")
    lines.append("### External benchmark review")
    for finding in external.get("findings", []):
        lines.append(f"- {traffic_icon(finding.get('traffic_light'))} **{finding.get('benchmark_area')}** — {finding.get('finding')} Allowed use: {finding.get('allowed_use')}")
    lines.append("")

    previous = report.get("previous_request_accommodation", {})
    prior = report.get("prior_deal_comparison", {})
    macro = report.get("macro_context_review", {})

    lines.append("## 11. Grey / Not Assessed SOW Modules")
    lines.append("")
    lines.append("| Module | Status | Reason | Required future data |")
    lines.append("|---|---|---|---|")
    for section in [previous, prior, macro]:
        lines.append(
            f"| {section.get('module')} | {traffic_icon(section.get('traffic_light'))} {section.get('status')} | "
            f"{clean_text(section.get('summary'))} | {clean_text(section.get('required_future_data'))} |"
        )
    lines.append("")

    open_items = report.get("conditions_precedent_open_items", {})
    lines.append("## 12. Conditions Precedent and Open Items")
    lines.append("")
    lines.append("| Category | Condition | Priority | Status | Sources |")
    lines.append("|---|---|---|---|---|")
    for row in open_items.get("open_items", []):
        lines.append(
            f"| {row.get('category')} | {row.get('condition')} | {row.get('priority')} | "
            f"{traffic_icon(row.get('traffic_light'))} {row.get('status')} | "
            f"{format_sources_md(row.get('source_chunks'))} |"
        )
    lines.append("")

    lines.append("## 13. Evidence Appendix")
    lines.append("")
    lines.append("| Chunk ID | Document ID | Corpus zone | Section | Source reference | Used in modules |")
    lines.append("|---|---|---|---|---|---|")
    for src in report.get("evidence_appendix", []):
        lines.append(
            f"| {src.get('chunk_id')} | {src.get('document_id')} | {src.get('corpus_zone')} | "
            f"{src.get('section_heading')} | {src.get('source_reference')} | "
            f"{', '.join(src.get('used_in_modules', []))} |"
        )
    lines.append("")

    lines.append("## 14. Audit Metadata")
    lines.append("")
    audit = report.get("audit_metadata", {})
    lines.append("```json")
    lines.append(json.dumps(audit, indent=2, ensure_ascii=False))
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Audit log and output persistence
# ---------------------------------------------------------------------

def write_audit_log(config_base, report):
    db_url = config_base.get("db_url")

    if not db_url:
        return {"status": "not_written", "reason": "db_url is not configured."}

    try:
        engine = create_engine(url=db_url, pool_pre_ping=True)

        create_sql = text("""
            CREATE TABLE IF NOT EXISTS report_generation_audit_log (
                audit_id BIGSERIAL PRIMARY KEY,
                run_id TEXT,
                transaction_id TEXT,
                report_type TEXT,
                report_version TEXT,
                classification TEXT,
                llm_model TEXT,
                generated_at TIMESTAMP WITH TIME ZONE,
                source_chunk_count INTEGER,
                report_json JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        insert_sql = text("""
            INSERT INTO report_generation_audit_log (
                run_id,
                transaction_id,
                report_type,
                report_version,
                classification,
                llm_model,
                generated_at,
                source_chunk_count,
                report_json
            )
            VALUES (
                :run_id,
                :transaction_id,
                :report_type,
                :report_version,
                :classification,
                :llm_model,
                :generated_at,
                :source_chunk_count,
                CAST(:report_json AS jsonb)
            );
        """)

        with engine.begin() as conn:
            conn.execute(create_sql)
            conn.execute(
                insert_sql,
                {
                    "run_id": report.get("run_id"),
                    "transaction_id": report.get("transaction_id"),
                    "report_type": report.get("report_type"),
                    "report_version": report.get("report_generation_version"),
                    "classification": report.get("classification"),
                    "llm_model": report.get("audit_metadata", {}).get("llm_model"),
                    "generated_at": report.get("generated_at"),
                    "source_chunk_count": len(report.get("evidence_appendix", [])),
                    "report_json": json.dumps(report, ensure_ascii=False),
                },
            )

        return {"status": "written", "table": "report_generation_audit_log"}

    except Exception as e:
        return {"status": "not_written", "reason": f"{type(e).__name__}: {str(e)[:1000]}"}


def save_report_outputs(config_pack, report, markdown_report):
    draft_report_dir = config_pack["draft_report_dir"]
    run_id = report["run_id"]

    json_path = draft_report_dir / f"{run_id}.json"
    markdown_path = draft_report_dir / f"{run_id}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


# ---------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------

def generate_investment_ic_review_report(
    transaction_id="TXN_HELIOS_001",
    use_llm_summary=True,
    write_audit=True,
):
    config_pack = get_config_pack()
    config_base = config_pack["config_base"]

    run_id = make_run_id(transaction_id)
    review_config = load_review_config()

    evidence_packets = run_retrieval_plan(transaction_id=transaction_id)

    transaction_profile = identify_transaction_profile(transaction_id, evidence_packets)

    completeness_readiness = run_completeness_check(review_config, evidence_packets)
    strategy_fit_assessment = run_strategy_fit_assessment(review_config, evidence_packets)
    historical_ic_question_coverage = run_historical_ic_question_check(review_config, evidence_packets)
    financial_reconciliation = run_financial_reconciliation(evidence_packets)
    market_offtake_revenue_review = run_market_offtake_revenue_review(evidence_packets)
    sensitivity_review = run_sensitivity_review(evidence_packets)
    risk_review = run_risk_review(review_config, evidence_packets)
    valuation_review = run_client_valuation_review(evidence_packets)
    external_benchmark_review = run_external_benchmark_review(evidence_packets)
    previous_request_accommodation = run_previous_request_accommodation_check(evidence_packets)
    prior_deal_comparison = run_prior_deal_comparison(evidence_packets)
    macro_context_review = run_macro_context_review(evidence_packets)
    conditions_precedent_open_items = build_open_items_register(evidence_packets)

    report_sections_for_dashboard = {
        "completeness_readiness": completeness_readiness,
        "strategy_fit_assessment": strategy_fit_assessment,
        "historical_ic_question_coverage": historical_ic_question_coverage,
        "financial_reconciliation": financial_reconciliation,
        "market_offtake_revenue_review": market_offtake_revenue_review,
        "sensitivity_review": sensitivity_review,
        "risk_review": risk_review,
        "valuation_review": valuation_review,
        "external_benchmark_review": external_benchmark_review,
        "previous_request_accommodation": previous_request_accommodation,
        "prior_deal_comparison": prior_deal_comparison,
        "macro_context_review": macro_context_review,
        "conditions_precedent_open_items": conditions_precedent_open_items,
    }

    executive_summary = build_executive_summary(report_sections_for_dashboard)
    traffic_light_dashboard = build_traffic_light_dashboard(report_sections_for_dashboard)
    evidence_appendix = build_evidence_appendix(evidence_packets)

    audit_metadata = {
        "run_id": run_id,
        "transaction_id": transaction_id,
        "report_type": REPORT_TYPE,
        "report_generation_version": REPORT_GENERATION_VERSION,
        "classification": CLASSIFICATION,
        "generated_at": utc_now_iso(),
        "llm_provider": config_base.get("llm_provider"),
        "llm_model": config_base.get("llm_model"),
        "retrieval_modules": list(evidence_packets.keys()),
        "source_chunk_count": len(evidence_appendix),
        "important_controls": [
            "No investment recommendation is made.",
            "Historical IC learning is not claimed if historical logs are unavailable.",
            "External EV/EBITDA benchmark support is not claimed unless evidence exists.",
            "Macro / FX / GDP conclusions are not fabricated where source data is unavailable.",
            "Every finding should be traceable through chunk_id/document_id/source_reference.",
        ],
    }

    report = {
        "run_id": run_id,
        "transaction_id": transaction_id,
        "report_type": REPORT_TYPE,
        "classification": CLASSIFICATION,
        "report_generation_version": REPORT_GENERATION_VERSION,
        "generated_at": audit_metadata["generated_at"],
        "transaction_profile": transaction_profile,
        "executive_summary": executive_summary,
        "traffic_light_dashboard": traffic_light_dashboard,
        "completeness_readiness": completeness_readiness,
        "strategy_fit_assessment": strategy_fit_assessment,
        "historical_ic_question_coverage": historical_ic_question_coverage,
        "financial_reconciliation": financial_reconciliation,
        "market_offtake_revenue_review": market_offtake_revenue_review,
        "sensitivity_review": sensitivity_review,
        "risk_review": risk_review,
        "valuation_review": valuation_review,
        "external_benchmark_review": external_benchmark_review,
        "previous_request_accommodation": previous_request_accommodation,
        "prior_deal_comparison": prior_deal_comparison,
        "macro_context_review": macro_context_review,
        "conditions_precedent_open_items": conditions_precedent_open_items,
        "evidence_appendix": evidence_appendix,
        "audit_metadata": audit_metadata,
    }

    if use_llm_summary:
        llm_summary = generate_llm_executive_summary(
            config_base=config_base,
            report_core={
                "executive_summary": executive_summary,
                "traffic_light_dashboard": traffic_light_dashboard,
                "completeness_readiness": completeness_readiness,
                "financial_reconciliation": financial_reconciliation,
                "market_offtake_revenue_review": market_offtake_revenue_review,
                "sensitivity_review": sensitivity_review,
                "risk_review": risk_review,
                "valuation_review": valuation_review,
                "external_benchmark_review": external_benchmark_review,
                "conditions_precedent_open_items": conditions_precedent_open_items,
                "grey_modules": {
                    "previous_request_accommodation": previous_request_accommodation,
                    "prior_deal_comparison": prior_deal_comparison,
                    "macro_context_review": macro_context_review,
                },
            },
        )
    else:
        llm_summary = {"status": "not_requested", "summary_text": ""}

    report["llm_executive_summary"] = llm_summary

    markdown_report = format_markdown_report(report)
    saved_outputs = save_report_outputs(config_pack, report, markdown_report)

    audit_write_result = {"status": "not_requested"}
    if write_audit:
        audit_write_result = write_audit_log(config_base, report)

    report["output_files"] = saved_outputs
    report["audit_write_result"] = audit_write_result

    markdown_report = format_markdown_report(report)
    saved_outputs = save_report_outputs(config_pack, report, markdown_report)
    report["output_files"] = saved_outputs

    return {
        "message": "AI first-line IC review report generated.",
        "status": "ok",
        "run_id": run_id,
        "transaction_id": transaction_id,
        "overall_readiness_status": executive_summary.get("overall_readiness_status"),
        "output_files": saved_outputs,
        "audit_write_result": audit_write_result,
        "report": report,
    }


if __name__ == "__main__":
    output = generate_investment_ic_review_report(
        transaction_id="TXN_HELIOS_001",
        use_llm_summary=True,
        write_audit=True,
    )

    print(json.dumps(
        {
            "message": output["message"],
            "status": output["status"],
            "run_id": output["run_id"],
            "transaction_id": output["transaction_id"],
            "overall_readiness_status": output["overall_readiness_status"],
            "output_files": output["output_files"],
            "audit_write_result": output["audit_write_result"],
        },
        indent=2,
        ensure_ascii=False,
    ))