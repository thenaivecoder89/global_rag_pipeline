# File_name: report_generation.py
# Purpose: Generate a SOW-driven AI first-line IC review pack for investment submissions.

# This is NOT a generic market research report generator.

# The script:
# 1. Uses retrieve_chunks.py for staged evidence retrieval
# 2. Builds SOW-aligned review modules
# 3. Clearly marks unavailable SOW areas as Grey / Not assessed
# 4. Produces JSON and Markdown outputs
# 5. Optionally writes an audit record to PostgreSQL

# Key controls:
# - No investment recommendation is made.
# - Historical IC learning is not claimed if historical logs are unavailable.
# - EV/EBITDA external benchmarking is not claimed unless external evidence exists.
# - Macro / FX / GDP conclusions are not fabricated where source data is unavailable.
# - Every finding is traceable through chunk_id / document_id / source_reference.

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

REPORT_GENERATION_VERSION = "report_generation_poc_v3"
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
        value = value.replace("X", "")
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
    seen_chunk_ids = set()

    for item in results:
        chunk_text = clean_text(item.get("chunk_text"))
        chunk_id = item.get("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue

        if has_any(chunk_text, terms):
            matched_sources.append(get_source_reference(item))
            seen_chunk_ids.add(chunk_id)

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


# ---------------------------------------------------------------------
# Source classification
# ---------------------------------------------------------------------

def classify_evidence_type(result):
    """
    Classify evidence using stable metadata only.

    Do not classify a chunk as workbook/memo/deck based on words inside chunk_text,
    because a deck can mention a workbook and a memo can mention a benchmark.
    """

    corpus_zone = clean_text(result.get("corpus_zone"))
    document_id = clean_text(result.get("document_id"))
    section_heading = normalize_text(result.get("section_heading"))
    source_reference = normalize_text(result.get("source_reference"))
    corpus_pack = normalize_text(result.get("corpus_pack"))

    if corpus_zone == "client_data":
        if document_id == "DOC_000016" or "synthetic_project_helios_financial_assumptions" in source_reference:
            return "client_workbook_control"

        if document_id == "DOC_000014":
            return "client_memo"

        if document_id == "DOC_000008" or section_heading.startswith("slide_"):
            return "client_deck"

        if document_id == "DOC_000001" or section_heading.startswith("page_"):
            return "client_pdf"

        if "docx_table" in source_reference or section_heading == "docx_document_text":
            return "client_supporting_note"

        return "client_evidence"

    if corpus_zone == "corpus_data":
        combined_metadata = normalize_text(
            f"{corpus_pack} {document_id} {section_heading} {source_reference}"
        )

        if "benchmark" in combined_metadata or "nrel" in combined_metadata or "utility-scale" in combined_metadata:
            return "external_benchmark"

        if "seed" in combined_metadata or "risk" in combined_metadata:
            return "risk_taxonomy"

        return "public_context"

    return "unknown"


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
    if evidence_type == "client_supporting_note":
        return "client_supporting_note"

    return "unknown"


# ---------------------------------------------------------------------
# Evidence selection scoring
# ---------------------------------------------------------------------

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
            "terms": ["merchant price exposure", "merchant exposure", "merchant price", "merchant curve"],
            "reviewer_challenge": "Has merchant revenue been supported by an independent price curve and downside confidence range?",
        },
        {
            "risk_id": "R02_GRID_COST_SCHEDULE",
            "risk": "Grid cost and schedule",
            "asset_class": "Solar / BESS",
            "terms": ["grid cost and schedule", "grid cost", "grid interconnection", "grid connection", "utility design"],
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
            "terms": ["offtaker credit", "offtaker", "payment security", "letter of credit", "counterparty"],
            "reviewer_challenge": "Has offtaker creditworthiness and payment security been independently verified?",
        },
        {
            "risk_id": "R05_LAND_PERMITTING",
            "risk": "Land, easements and permits",
            "asset_class": "Renewables",
            "terms": ["land and permits", "land", "easement", "permit", "permitting"],
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
                "max_chunk_chars": 12000,
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
                "max_chunk_chars": 12000,
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
# Table row parsing utilities
# ---------------------------------------------------------------------

def extract_row_texts(chunk_text):
    """
    Returns row text strings without the leading 'Row n:' label.
    Handles rows embedded in one long chunk.
    """

    chunk_text = clean_text(chunk_text)
    pattern = r"Row\s+\d+:\s*(.*?)(?=\s+Row\s+\d+:|$)"
    rows = re.findall(pattern, chunk_text, flags=re.IGNORECASE)

    return [clean_text(row) for row in rows]


def parse_table_row_cells(row_text):
    """
    Parses rows like:
    0: Total project cost | 1: 289.0 | 2: USD mn
    Unnamed: 0: Total project cost | Unnamed: 1: 289.0 | Unnamed: 2: USD mn

    Returns:
    [
      {"label": "0", "value": "Total project cost"},
      {"label": "1", "value": "289.0"}
    ]
    """

    cells = []
    parts = [part.strip() for part in row_text.split("|")]

    for part in parts:
        match = re.match(r"(?:Unnamed:\s*)?([^:]+):\s*(.*)$", part, flags=re.IGNORECASE)

        if match:
            label = clean_text(match.group(1))
            value = clean_text(match.group(2))
        else:
            label = ""
            value = clean_text(part)

        cells.append({"label": label, "value": value})

    return cells


def cell_values_from_row(row_text):
    cells = parse_table_row_cells(row_text)
    return [cell["value"] for cell in cells]


def get_cell_value_by_label(row_text, label_candidates):
    """Return the value for the first matching parsed table label."""

    cells = parse_table_row_cells(row_text)
    normalized_candidates = [normalize_text(label) for label in label_candidates]

    for cell in cells:
        label = normalize_text(cell.get("label"))

        for candidate in normalized_candidates:
            if label == candidate or candidate in label:
                return clean_text(cell.get("value"))

    return None


def row_source_is_table(result):
    source_reference = normalize_text(result.get("source_reference"))
    section_heading = normalize_text(result.get("section_heading"))
    return "extracted_table_rows" in source_reference or section_heading.startswith("table:")


def first_number_in_text(text_value):
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", clean_text(text_value))
    if match:
        return safe_float(match.group(0))
    return None


# ---------------------------------------------------------------------
# Completeness and readiness
# ---------------------------------------------------------------------

def run_completeness_check(review_config, evidence_packets):
    checklist = review_config["completeness_checklist"]
    evidence_results = evidence_packets.get("completeness_readiness", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)

    external_cost_blob = evidence_packets.get("external_cost_benchmark", {}).get("evidence_blob", "")
    external_ev_blob = evidence_packets.get("external_ev_ebitda_benchmark", {}).get("evidence_blob", "")

    external_cost_available = has_any(
        external_cost_blob,
        ["utility-scale battery storage", "utility-scale pv", "pv-plus-battery", "overnight capital cost", "grid connection cost"],
    )

    external_ev_available = has_all(external_ev_blob, ["ev", "ebitda"]) and has_any(
        external_ev_blob,
        ["enterprise value", "valuation multiple", "transaction benchmark", "comparable transaction", "listed company"],
    )

    rows = []

    for item in checklist:
        evidence_found = has_any(blob, item["evidence_terms"])
        weak_evidence = has_any(blob, item["weak_terms"])

        if item["item_id"] == "C12_BENCHMARK_SUPPORT":
            if external_cost_available and external_ev_available:
                status = READINESS_PASS
                traffic_light = STATUS_GREEN
                weakness = "External cost and EV/EBITDA benchmark evidence found."
                evidence_found = True
            elif external_cost_available and not external_ev_available:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "External cost benchmark evidence is available, but external EV/EBITDA valuation multiple evidence is not available."
                evidence_found = True
            elif evidence_found:
                status = READINESS_WEAK
                traffic_light = STATUS_AMBER
                weakness = "Only client-side benchmark assertions are evidenced; independent benchmark support is incomplete."
            else:
                status = READINESS_MISSING
                traffic_light = STATUS_RED
                weakness = "Benchmark support not found in retrieved evidence."
        else:
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

        if item["item_id"] == "C12_BENCHMARK_SUPPORT" and external_cost_available:
            external_sources = evidence_packets.get("external_cost_benchmark", {}).get("selected_sources", [])
            source_chunks = source_chunks + external_sources[:3]

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


# ---------------------------------------------------------------------
# Strategy and fit assessment
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Historical IC question coverage
# ---------------------------------------------------------------------

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
            source_chunks = find_source_chunks(
                results=current_submission_results,
                terms=question["coverage_terms"],
                max_sources=5,
            )
            negative_evidence_chunks = []
        elif coverage_found and weak_found:
            coverage_status = "Partially addressed"
            traffic_light = STATUS_AMBER
            source_chunks = find_source_chunks(
                results=current_submission_results,
                terms=question["coverage_terms"] + question["weak_terms"],
                max_sources=5,
            )
            negative_evidence_chunks = find_source_chunks(
                results=current_submission_results,
                terms=question["weak_terms"],
                max_sources=3,
            )
        else:
            coverage_status = "Not addressed"
            traffic_light = STATUS_RED
            source_chunks = []
            negative_evidence_chunks = []

        rows.append(
            {
                "question_id": question["question_id"],
                "theme": question["theme"],
                "likely_ic_question": question["question"],
                "coverage_status": coverage_status,
                "traffic_light": traffic_light,
                "source_chunks": source_chunks,
                "negative_evidence_chunks": negative_evidence_chunks,
            }
        )

    return {
        "module": "Historical IC Question Coverage",
        "traffic_light": STATUS_AMBER,
        "summary": "Historical IC logs are not evidenced; standard IC question library is used for this POC run.",
        "historical_logs_available": historical_logs_available,
        "limitation": (
            "Historical IC Q&A logs / follow-up actions were not clearly evidenced in the current corpus. "
            "For this POC run, the tool uses a standard IC question library and does not claim to have learned "
            "from actual historical IC interactions."
        ),
        "question_coverage": rows,
    }


# ---------------------------------------------------------------------
# Financial metric extraction and reconciliation
# ---------------------------------------------------------------------

def metric_value_is_plausible(metric_key, value):
    if value is None:
        return False

    if metric_key == "total_project_cost_usd_mn":
        return 100 <= value <= 2000

    if metric_key == "project_irr_pct":
        return 3 <= value <= 50

    if metric_key == "equity_irr_pct":
        return 3 <= value <= 60

    if metric_key == "npv_usd_mn":
        return -1000 <= value <= 1000 and abs(value) >= 3

    if metric_key == "minimum_dscr_x":
        return 0.5 <= value <= 3.5

    if metric_key == "ev_ebitda_x":
        return 2.5 <= value <= 30

    if metric_key == "debt_total_cost_pct":
        return 0 <= value <= 100

    if metric_key == "year_one_revenue_usd_mn":
        return 1 <= value <= 1000

    if metric_key == "ppa_price_usd_mwh":
        return 10 <= value <= 250

    if metric_key == "capacity_factor_pct":
        return 5 <= value <= 70

    return True


def get_metric_definitions():
    return [
        {
            "metric_key": "total_project_cost_usd_mn",
            "metric_name": "Total project cost",
            "terms": ["total project cost"],
            "unit": "USD mn",
            "direct_patterns": [
                r"total project cost[^0-9]{0,50}(?:usd\s*)?([0-9]+(?:\.[0-9]+)?)",
            ],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "project_irr_pct",
            "metric_name": "Project IRR",
            "terms": ["project irr"],
            "unit": "%",
            "direct_patterns": [r"project irr[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "equity_irr_pct",
            "metric_name": "Equity IRR",
            "terms": ["equity irr"],
            "unit": "%",
            "direct_patterns": [r"equity irr[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "npv_usd_mn",
            "metric_name": "NPV",
            "terms": ["npv"],
            "unit": "USD mn",
            "direct_patterns": [r"\bnpv\b[^0-9\-]{0,50}(?:usd\s*)?([-+]?[0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "minimum_dscr_x",
            "metric_name": "Minimum DSCR",
            "terms": ["minimum dscr", "min_dscr", "min dscr"],
            "unit": "x",
            "direct_patterns": [r"(?:minimum dscr|min_dscr|min dscr)[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "ev_ebitda_x",
            "metric_name": "EV / EBITDA",
            "terms": ["ev / ebitda", "ev ebitda", "ev/ebitda"],
            "unit": "x",
            "direct_patterns": [r"ev\s*/?\s*ebitda[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "debt_total_cost_pct",
            "metric_name": "Debt / total cost",
            "terms": ["debt / total cost", "debt total cost", "debt-to-cost"],
            "unit": "%",
            "direct_patterns": [r"(?:debt\s*/\s*total cost|debt total cost|debt-to-cost)[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "year_one_revenue_usd_mn",
            "metric_name": "Year-one revenue",
            "terms": ["year-one revenue", "year one revenue"],
            "unit": "USD mn",
            "direct_patterns": [r"(?:year-one revenue|year one revenue)[^0-9]{0,50}(?:usd\s*)?([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions", "docx_table_4"],
        },
        {
            "metric_key": "ppa_price_usd_mwh",
            "metric_name": "PPA price",
            "terms": ["ppa price", "contracted ppa price"],
            "unit": "USD/MWh",
            "direct_patterns": [
                r"ppa price[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)",
                r"usd\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*mwh",
            ],
            "allow_table_names": ["summary", "assumptions"],
        },
        {
            "metric_key": "capacity_factor_pct",
            "metric_name": "Capacity factor",
            "terms": ["capacity factor", "p50 capacity factor"],
            "unit": "%",
            "direct_patterns": [r"capacity factor[^0-9]{0,50}([0-9]+(?:\.[0-9]+)?)"],
            "allow_table_names": ["summary", "assumptions"],
        },
    ]


def metric_source_allowed(result, metric):
    """Restrict metric parsing to proper financial-metric sources."""

    source_label = infer_source_label(result)
    section_heading = normalize_text(result.get("section_heading"))
    source_reference = normalize_text(result.get("source_reference"))
    document_id = clean_text(result.get("document_id"))

    if source_label not in ["workbook", "memo", "deck", "pdf_deck"]:
        return False

    if metric["metric_key"] == "total_project_cost_usd_mn":
        if "project costs" in section_heading and "summary" not in section_heading:
            return False

    if document_id not in ["DOC_000016", "DOC_000014", "DOC_000008", "DOC_000001"]:
        return False

    allowed_names = metric.get("allow_table_names", [])

    if source_label in ["workbook", "memo"] and row_source_is_table(result):
        if not any(name in section_heading or name in source_reference for name in allowed_names):
            return False

    return True


def extract_metric_from_table_row(row_text, metric):
    row_lower = normalize_text(row_text)

    if not any(term in row_lower for term in metric["terms"]):
        return None

    if metric["metric_key"] == "total_project_cost_usd_mn" and "total project cost" not in row_lower:
        return None

    cells = parse_table_row_cells(row_text)
    values = [cell["value"] for cell in cells]

    metric_cell_index = None

    for index, value in enumerate(values):
        if any(term in normalize_text(value) for term in metric["terms"]):
            metric_cell_index = index
            break

    if metric_cell_index is None:
        return None

    candidate_values = []

    for value in values[metric_cell_index + 1:]:
        number_value = first_number_in_text(value)

        if number_value is not None:
            candidate_values.append(number_value)

    for candidate in candidate_values:
        if metric_value_is_plausible(metric["metric_key"], candidate):
            return candidate

    return None


def extract_metric_from_direct_text(chunk_text, metric):
    text_value = clean_text(chunk_text)

    if re.search(r"Row\s+\d+:", text_value, flags=re.IGNORECASE):
        return None

    for pattern in metric["direct_patterns"]:
        match = re.search(pattern, text_value, flags=re.IGNORECASE)

        if not match:
            continue

        candidate = safe_float(match.group(1))

        if metric_value_is_plausible(metric["metric_key"], candidate):
            return candidate

    return None


def extract_financial_metrics(evidence_results):
    metric_definitions = get_metric_definitions()
    rows = []
    seen = set()

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        source_label = infer_source_label(result)

        if source_label not in ["workbook", "memo", "deck", "pdf_deck"]:
            continue

        row_texts = extract_row_texts(chunk_text)

        for metric in metric_definitions:
            if not metric_source_allowed(result, metric):
                continue

            for row_text in row_texts:
                value = extract_metric_from_table_row(row_text, metric)

                if value is None:
                    continue

                signature = (metric["metric_key"], source_label, round(value, 6), result.get("chunk_id"), "table_row")

                if signature in seen:
                    continue

                seen.add(signature)

                rows.append({
                    "metric_key": metric["metric_key"],
                    "metric_name": metric["metric_name"],
                    "source_label": source_label,
                    "value": value,
                    "unit": metric["unit"],
                    "source_chunk": get_source_reference(result),
                    "extraction_method": "table_row",
                })

            value = extract_metric_from_direct_text(chunk_text, metric)

            if value is None:
                continue

            signature = (metric["metric_key"], source_label, round(value, 6), result.get("chunk_id"), "direct_text")

            if signature in seen:
                continue

            seen.add(signature)

            rows.append({
                "metric_key": metric["metric_key"],
                "metric_name": metric["metric_name"],
                "source_label": source_label,
                "value": value,
                "unit": metric["unit"],
                "source_chunk": get_source_reference(result),
                "extraction_method": "direct_text",
            })

    return rows


def unique_float_values(values, tolerance=0.0001):
    unique_values = []
    for value in values:
        if value is None:
            continue
        already_seen = False
        for existing in unique_values:
            if abs(float(existing) - float(value)) <= tolerance:
                already_seen = True
                break
        if not already_seen:
            unique_values.append(value)
    return unique_values


def dedupe_source_chunks(chunks):
    output = []
    seen = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        key = (chunk.get("chunk_id"), chunk.get("source_reference"))
        if key in seen:
            continue
        seen.add(key)
        output.append(chunk)
    return output


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
                "source_chunks_by_source": {},
            }
        source_label = row["source_label"]
        by_metric[metric_key]["values_by_source"].setdefault(source_label, []).append(row["value"])
        by_metric[metric_key]["source_chunks_by_source"].setdefault(source_label, []).append(row["source_chunk"])

    reconciliation_rows = []
    for metric_key, item in by_metric.items():
        cleaned_values_by_source = {}
        source_chunks = []
        for source_label, values in item["values_by_source"].items():
            cleaned_values_by_source[source_label] = unique_float_values(values)
        for chunks in item["source_chunks_by_source"].values():
            source_chunks.extend(chunks)
        source_chunks = dedupe_source_chunks(source_chunks)

        all_values = []
        for values in cleaned_values_by_source.values():
            all_values.extend(values)
        distinct_values = unique_float_values(all_values)

        preferred_value = None
        preferred_source = None
        for source_label in ["workbook", "memo", "deck", "pdf_deck", "unknown"]:
            source_values = cleaned_values_by_source.get(source_label, [])
            if source_values:
                preferred_value = source_values[0]
                preferred_source = source_label
                break

        source_level_conflict = any(len(values) > 1 for values in cleaned_values_by_source.values())
        if len(distinct_values) > 1:
            traffic_light = STATUS_RED
            issue = "Inconsistent values found across retrieved sources."
        elif source_level_conflict:
            traffic_light = STATUS_RED
            issue = "Multiple values found within at least one source category."
        elif len(distinct_values) == 1:
            traffic_light = STATUS_GREEN
            issue = "No inconsistency detected across retrieved values."
        else:
            traffic_light = STATUS_GREY
            issue = "No numeric value extracted."

        reconciliation_rows.append({
            "metric_key": metric_key,
            "metric_name": item["metric_name"],
            "unit": item["unit"],
            "values_by_source": cleaned_values_by_source,
            "preferred_value_for_review": preferred_value,
            "preferred_source": preferred_source,
            "traffic_light": traffic_light,
            "issue": issue,
            "source_chunks": source_chunks,
        })

    metric_order = [
        "total_project_cost_usd_mn", "project_irr_pct", "equity_irr_pct", "npv_usd_mn",
        "minimum_dscr_x", "ev_ebitda_x", "debt_total_cost_pct", "year_one_revenue_usd_mn",
        "ppa_price_usd_mwh", "capacity_factor_pct",
    ]
    order_map = {metric_key: index for index, metric_key in enumerate(metric_order)}
    return sorted(reconciliation_rows, key=lambda row: order_map.get(row["metric_key"], 999))


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
        "summary": "Financial metrics are extracted using row-aware parsing and reconciled across workbook, memo, deck and PDF evidence.",
        "metric_rows_extracted": metric_rows,
        "reconciliation_table": reconciliation_rows,
    }


# ---------------------------------------------------------------------
# Sensitivity review
# ---------------------------------------------------------------------

def get_sensitivity_scenarios():
    return [
        "Base Case",
        "Merchant price -15%",
        "Generation -5%",
        "Capex +10%",
        "COD delay 6 months",
        "FX depreciation 10%",
        "Combined downside",
        "BESS degradation / augmentation",
    ]


def normalize_sensitivity_scenario(value):
    value_norm = normalize_text(value)
    mapping = {
        "base case": "Base Case",
        "merchant price -15%": "Merchant price -15%",
        "generation -5%": "Generation -5%",
        "capex +10%": "Capex +10%",
        "cod delay 6 months": "COD delay 6 months",
        "fx depreciation 10%": "FX depreciation 10%",
        "combined downside": "Combined downside",
        "bess degradation / augmentation": "BESS degradation / augmentation",
        "battery augmentation": "BESS degradation / augmentation",
    }
    for key, label in mapping.items():
        if key in value_norm:
            return label
    return None


def extract_workbook_sensitivity_case(row_text, result):
    scenario = get_cell_value_by_label(row_text, ["Sensitivity Cases"])
    scenario = normalize_sensitivity_scenario(scenario)
    if not scenario:
        return None
    equity_irr = safe_float(get_cell_value_by_label(row_text, ["6", "Equity_IRR_pct"]))
    minimum_dscr = safe_float(get_cell_value_by_label(row_text, ["7", "Min_DSCR_x"]))
    included_flag = clean_text(get_cell_value_by_label(row_text, ["8", "Included_In_Submission"]))
    if included_flag.lower() == "yes":
        included_flag = "Yes"
    elif included_flag.lower() == "no":
        included_flag = "No"
    else:
        included_flag = None
    if equity_irr is None or minimum_dscr is None:
        return None
    return {
        "scenario": scenario,
        "equity_irr_pct": equity_irr,
        "minimum_dscr_x": minimum_dscr,
        "included_in_submission": included_flag,
        "source_label": infer_source_label(result),
        "source_chunk": get_source_reference(result),
        "extraction_method": "workbook_labelled_table",
    }


def extract_simple_sensitivity_case(row_text, result):
    cells = parse_table_row_cells(row_text)
    values = [cell["value"] for cell in cells]
    scenario = None
    scenario_index = None
    for index, value in enumerate(values):
        scenario = normalize_sensitivity_scenario(value)
        if scenario:
            scenario_index = index
            break
    if not scenario:
        return None
    if has_any(row_text, ["not shown", "not included", "does not include"]):
        return None
    candidate_numbers = []
    for value in values[scenario_index + 1:]:
        number_value = first_number_in_text(value)
        if number_value is not None:
            candidate_numbers.append(number_value)
    if len(candidate_numbers) < 1:
        return None
    equity_irr = None
    minimum_dscr = None
    for number_value in candidate_numbers:
        if equity_irr is None and 3 <= number_value <= 60:
            equity_irr = number_value
            continue
        if minimum_dscr is None and 0.5 <= number_value <= 3.5:
            minimum_dscr = number_value
            continue
    return {
        "scenario": scenario,
        "equity_irr_pct": equity_irr,
        "minimum_dscr_x": minimum_dscr,
        "included_in_submission": None,
        "source_label": infer_source_label(result),
        "source_chunk": get_source_reference(result),
        "extraction_method": "simple_table_row",
    }


def extract_sensitivity_cases(evidence_results):
    sensitivity_cases = []
    seen = set()
    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        row_texts = extract_row_texts(chunk_text)
        for row_text in row_texts:
            if not has_any(row_text, get_sensitivity_scenarios()):
                continue
            case = None
            if "synthetic_project_helios_financial_assumptions_sensitivities" in normalize_text(result.get("section_heading")):
                case = extract_workbook_sensitivity_case(row_text, result)
            if case is None:
                case = extract_simple_sensitivity_case(row_text, result)
            if case is None:
                continue
            signature = (case.get("scenario"), case.get("equity_irr_pct"), case.get("minimum_dscr_x"), case.get("included_in_submission"), case.get("source_chunk", {}).get("chunk_id"))
            if signature in seen:
                continue
            seen.add(signature)
            sensitivity_cases.append(case)
    return sensitivity_cases


def dedupe_sensitivity_cases(cases):
    by_scenario = {}
    for case in cases:
        scenario = clean_text(case.get("scenario"))
        if not scenario:
            continue
        quality_score = 0
        if case.get("source_label") == "workbook":
            quality_score += 10
        if case.get("included_in_submission") in ["Yes", "No"]:
            quality_score += 6
        if case.get("equity_irr_pct") is not None:
            quality_score += 2
        if case.get("minimum_dscr_x") is not None:
            quality_score += 2
        if case.get("extraction_method") == "workbook_labelled_table":
            quality_score += 4
        if scenario not in by_scenario or quality_score > by_scenario[scenario]["quality_score"]:
            by_scenario[scenario] = {"quality_score": quality_score, "case": case}
    output_cases = []
    for scenario in get_sensitivity_scenarios():
        if scenario in by_scenario:
            output_cases.append(by_scenario[scenario]["case"])
    for scenario, data in by_scenario.items():
        if scenario not in get_sensitivity_scenarios():
            output_cases.append(data["case"])
    return output_cases


def extract_submitted_pack_sensitivity_gaps(evidence_results):
    gaps = []
    seen = set()
    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        if has_any(chunk_text, ["combined downside not shown", "does not include a combined downside case"]):
            key = "combined_downside_not_shown"
            if key not in seen:
                gaps.append({"gap": "Combined downside not shown in submitted pack.", "source_chunk": get_source_reference(result)})
                seen.add(key)
        if has_any(chunk_text, ["battery augmentation sensitivity not shown", "bess degradation/augmentation sensitivity", "bess degradation and augmentation sensitivity"]):
            key = "bess_augmentation_not_shown"
            if key not in seen:
                gaps.append({"gap": "BESS degradation / augmentation sensitivity not shown in submitted pack.", "source_chunk": get_source_reference(result)})
                seen.add(key)
    return gaps


def validate_sensitivity_cases(cases):
    expected = {
        "Base Case": (13.9, 1.32, "Yes"),
        "Merchant price -15%": (12.2, 1.24, "Yes"),
        "Generation -5%": (12.6, 1.21, "Yes"),
        "Capex +10%": (11.8, 1.30, "Yes"),
        "COD delay 6 months": (11.9, 1.28, "Yes"),
        "FX depreciation 10%": (12.7, 1.27, "Yes"),
        "Combined downside": (8.5, 1.08, "No"),
        "BESS degradation / augmentation": (12.8, 1.26, "No"),
    }
    failures = []
    case_map = {case.get("scenario"): case for case in cases}
    for scenario, expected_values in expected.items():
        expected_equity_irr, expected_dscr, expected_flag = expected_values
        case = case_map.get(scenario)
        if not case:
            failures.append(f"Missing sensitivity case: {scenario}")
            continue
        actual_equity_irr = case.get("equity_irr_pct")
        actual_dscr = case.get("minimum_dscr_x")
        actual_flag = case.get("included_in_submission")
        if actual_equity_irr is None or abs(float(actual_equity_irr) - expected_equity_irr) > 0.01:
            failures.append(f"Incorrect equity IRR for {scenario}: {actual_equity_irr}")
        if actual_dscr is None or abs(float(actual_dscr) - expected_dscr) > 0.01:
            failures.append(f"Incorrect minimum DSCR for {scenario}: {actual_dscr}")
        if actual_flag != expected_flag:
            failures.append(f"Incorrect included_in_submission flag for {scenario}: {actual_flag}")
    return failures


def run_sensitivity_review(evidence_packets):
    evidence_results = evidence_packets.get("sensitivity_review", {}).get("selected_results", [])
    blob = get_evidence_blob(evidence_results)
    raw_sensitivity_cases = extract_sensitivity_cases(evidence_results)
    sensitivity_cases = dedupe_sensitivity_cases(raw_sensitivity_cases)
    submitted_pack_gaps = extract_submitted_pack_sensitivity_gaps(evidence_results)
    validation_failures = validate_sensitivity_cases(sensitivity_cases)
    combined_downside_available = any(case.get("scenario") == "Combined downside" for case in sensitivity_cases)
    bess_augmentation_available = any(case.get("scenario") == "BESS degradation / augmentation" for case in sensitivity_cases)
    submission_gap_flag = any(case.get("scenario") in ["Combined downside", "BESS degradation / augmentation"] and case.get("included_in_submission") == "No" for case in sensitivity_cases)
    if validation_failures:
        module_status = STATUS_RED
    elif submission_gap_flag:
        module_status = STATUS_AMBER
    elif not sensitivity_cases and not has_any(blob, ["sensitivity"]):
        module_status = STATUS_RED
    else:
        module_status = STATUS_GREEN
    return {
        "module": "Sensitivity and Downside Protection Review",
        "traffic_light": module_status,
        "summary": "Sensitivity review distinguishes submitted cases from workbook-only cases and validates workbook sensitivity values.",
        "combined_downside_available_in_evidence": combined_downside_available,
        "bess_augmentation_available_in_evidence": bess_augmentation_available,
        "submission_gap_flag": submission_gap_flag,
        "sensitivity_cases": sensitivity_cases,
        "submitted_pack_gaps": submitted_pack_gaps,
        "validation_failures": validation_failures,
        "raw_case_count_before_deduplication": len(raw_sensitivity_cases),
        "deduped_case_count": len(sensitivity_cases),
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
        "merchant_curve_missing": has_any(blob, ["merchant curve source not appended", "independent merchant curve referenced but not appended", "independent merchant-price study is referenced but not included", "independent curve absent"]),
        "offtaker_credit_missing": has_any(blob, ["no independent credit", "offtaker-credit support are missing", "offtaker credit support are missing", "independent credit review absent"]),
        "payment_security_pending": has_any(blob, ["payment-security package remains under negotiation", "payment security", "under discussion", "under negotiation", "not executed"]),
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
                "source_chunks": find_source_chunks(evidence_results, ["merchant curve", "not appended", "not included", "independent curve absent"]),
            }
        )

    if flags["offtaker_credit_missing"] or flags["payment_security_pending"]:
        findings.append(
            {
                "finding": "Offtaker credit and payment-security package are not fully evidenced.",
                "traffic_light": STATUS_RED,
                "explanation": "Payment security / credit support appears under discussion, not fully executed or independently reviewed.",
                "source_chunks": find_source_chunks(evidence_results, ["offtaker", "credit", "payment security", "under negotiation", "under discussion", "not executed"]),
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
        "summary": "Market/offtake review flags merchant curve, offtaker credit, payment security and PPA price consistency issues.",
        "flags": flags,
        "findings": findings,
    }


# ---------------------------------------------------------------------
# Risk register
# ---------------------------------------------------------------------

def extract_risk_table_rows(evidence_results):
    risk_rows = []

    for result in evidence_results:
        chunk_text = clean_text(result.get("chunk_text"))
        row_texts = extract_row_texts(chunk_text)

        for row_text in row_texts:
            values = cell_values_from_row(row_text)

            if len(values) < 2:
                continue

            first_cell = clean_text(values[0])
            second_cell = clean_text(values[1])

            if normalize_text(first_cell) in ["risk", "principal risks"]:
                continue

            # Risk table rows generally have:
            # cell 0 = risk, cell 1 = rating/assessment, cell 2 = mitigation/status
            if has_any(second_cell, ["high", "medium", "low", "not assessed"]):
                mitigation_status = values[2] if len(values) >= 3 else ""

                risk_rows.append(
                    {
                        "risk_text": first_cell,
                        "rating": second_cell,
                        "mitigation_status": clean_text(mitigation_status),
                        "source_chunk": get_source_reference(result),
                    }
                )

    return risk_rows


def find_best_risk_row(risk_rows, terms):
    for row in risk_rows:
        if has_any(row.get("risk_text"), terms):
            return row

    for row in risk_rows:
        if has_any(row.get("mitigation_status"), terms):
            return row

    return None


def risk_rating_to_traffic_light(rating, mitigation_status):
    rating_lower = normalize_text(rating)
    mitigation_lower = normalize_text(mitigation_status)

    if "not assessed" in rating_lower:
        return STATUS_RED

    if "not assessed" in mitigation_lower or "no dedicated" in mitigation_lower or "no funded" in mitigation_lower:
        return STATUS_RED

    if "high" in rating_lower:
        return STATUS_AMBER

    if "medium-high" in rating_lower:
        return STATUS_AMBER

    if "medium" in rating_lower:
        return STATUS_AMBER

    if "low" in rating_lower:
        return STATUS_GREEN

    return STATUS_GREY


def run_risk_review(review_config, evidence_packets):
    risk_taxonomy = review_config["asset_class_risk_taxonomy"]
    evidence_results = evidence_packets.get("risk_review", {}).get("selected_results", [])
    risk_rows = extract_risk_table_rows(evidence_results)

    rows = []

    for risk_item in risk_taxonomy:
        best_row = find_best_risk_row(risk_rows, risk_item["terms"])

        if not best_row:
            rows.append(
                {
                    "risk_id": risk_item["risk_id"],
                    "risk": risk_item["risk"],
                    "asset_class": risk_item["asset_class"],
                    "rating": "Not identified",
                    "mitigation_status": "No row-level evidence found in retrieved risk tables.",
                    "reviewer_challenge": risk_item["reviewer_challenge"],
                    "open_gap": "Risk topic not clearly addressed in retrieved evidence.",
                    "traffic_light": STATUS_GREY,
                    "source_chunks": [],
                }
            )
            continue

        rating = best_row.get("rating")
        mitigation_status = best_row.get("mitigation_status")
        traffic_light = risk_rating_to_traffic_light(rating, mitigation_status)

        rows.append(
            {
                "risk_id": risk_item["risk_id"],
                "risk": risk_item["risk"],
                "asset_class": risk_item["asset_class"],
                "rating": rating,
                "mitigation_status": mitigation_status,
                "reviewer_challenge": risk_item["reviewer_challenge"],
                "open_gap": "Review the mitigation/status evidence and request support where weak language is present.",
                "traffic_light": traffic_light,
                "source_chunks": [best_row.get("source_chunk")],
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
        "summary": "Risk register is built from row-level risk table parsing to avoid contamination across unrelated risk rows.",
        "risk_register": rows,
        "raw_risk_rows_extracted": risk_rows,
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
        "summary": "Client-side valuation review separates reported valuation metrics from independent benchmark confirmation.",
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
        "summary": "External benchmark review allows cost benchmarking but blocks unsupported EV/EBITDA benchmark claims.",
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
        "summary": "Open items are categorized into approval-readiness conditions for reviewer follow-up.",
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
# Terms and definitions
# ---------------------------------------------------------------------

def build_terms_and_definitions():
    terms = [
        ("AI", "Artificial intelligence; software that can perform tasks that normally require human reasoning or pattern recognition."),
        ("API", "Application programming interface; a structured way for software systems to call each other."),
        ("Amber", "Traffic-light status meaning partly addressed, weakly evidenced or requiring reviewer follow-up."),
        ("Audit log", "A record of the report run, model used, source chunks used and output generated."),
        ("BESS", "Battery Energy Storage System; batteries used to store electricity and release it later."),
        ("Benchmark", "A comparison point used to assess whether a value, assumption or cost is reasonable."),
        ("Brownfield", "An investment in an existing or operating asset, or expansion of an existing project."),
        ("Capex", "Capital expenditure; upfront investment cost to build or acquire an asset."),
        ("COD", "Commercial Operations Date; the date on which the project is expected to start commercial operation."),
        ("Contracted revenue", "Revenue backed by a contract such as a PPA rather than exposed to market prices."),
        ("Curtailment", "Reduction in electricity generation because the grid or buyer cannot accept all available output."),
        ("Data room", "A structured repository of transaction documents used for diligence and review."),
        ("Decommissioning", "Activities and costs required to dismantle an asset at the end of its useful life."),
        ("Debt / total cost", "Debt funding as a percentage of total project cost."),
        ("DSCR", "Debt Service Coverage Ratio; cash available for debt service divided by required debt payments."),
        ("EBITDA", "Earnings before interest, taxes, depreciation and amortization; a common operating-profit measure."),
        ("EMS", "Energy Management System; software used to monitor and optimize energy assets such as BESS."),
        ("Equity IRR", "Internal rate of return to equity investors after debt financing effects."),
        ("ESG", "Environmental, social and governance considerations."),
        ("EV / EBITDA", "Enterprise value divided by EBITDA; a valuation multiple."),
        ("External benchmark", "Benchmark evidence from outside the submitted transaction pack."),
        ("FX", "Foreign exchange; currency movement or exchange-rate exposure."),
        ("GDP", "Gross domestic product; a measure of economic output."),
        ("Green", "Traffic-light status meaning adequately addressed with clear evidence."),
        ("Greenfield", "A project developed and built from a new site or early-stage development base."),
        ("Grey", "Traffic-light status meaning not assessable because source data or configuration is unavailable."),
        ("Grid interconnection", "The connection between the project and the electricity grid."),
        ("IC", "Investment Committee; governance body that reviews and approves investment proposals."),
        ("IC memo", "Investment Committee memorandum; the core written submission for investment review."),
        ("IRR", "Internal Rate of Return; discount rate at which projected cash flows produce zero net present value."),
        ("JSON", "JavaScript Object Notation; structured data format used by APIs and front ends."),
        ("Letter of credit", "Bank-backed payment-security instrument used to support counterparty obligations."),
        ("LLM", "Large language model; AI model used to read, summarize and generate text."),
        ("Macro context", "Macroeconomic context such as GDP growth, inflation, exchange rates and country risk."),
        ("Majority stake", "Ownership interest above 50%, usually giving control or strong governance influence."),
        ("Merchant exposure", "Revenue exposure to market prices rather than fixed contracted prices."),
        ("MWh", "Megawatt-hour; unit of energy equal to one megawatt generated or consumed for one hour."),
        ("MW", "Megawatt; unit of power capacity."),
        ("MWac", "Megawatt alternating-current capacity; grid-side power capacity measure."),
        ("MWp", "Megawatt-peak; peak solar panel capacity under standard test conditions."),
        ("NPV", "Net Present Value; present value of expected cash flows minus investment cost."),
        ("O&M", "Operations and maintenance costs incurred to operate an asset."),
        ("Offtake", "Sale or purchase arrangement for the electricity or output from a project."),
        ("Offtaker", "Counterparty that buys the project output under an offtake agreement."),
        ("Payment security", "Contractual or bank-backed protection that supports payment obligations."),
        ("PPA", "Power Purchase Agreement; contract for sale and purchase of electricity."),
        ("POC", "Proof of concept; early version built to validate feasibility."),
        ("Project IRR", "Internal rate of return of the project before equity financing effects."),
        ("PV", "Photovoltaic solar technology that converts sunlight into electricity."),
        ("RAG", "Retrieval-augmented generation; AI approach that retrieves source evidence before generating answers."),
        ("Red", "Traffic-light status meaning missing, contradictory or approval-critical issue."),
        ("Reserved matters", "Important decisions requiring specified shareholder or board approval."),
        ("SCADA", "Supervisory Control and Data Acquisition; system used to monitor and control technical assets."),
        ("Sensitivity analysis", "Testing how investment outputs change when key assumptions change."),
        ("SharePoint", "Document management platform where source files may be stored."),
        ("Source chunk", "A retrieved text or table segment used as evidence for a finding."),
        ("SOW", "Statement of Work; document defining the client requirements and scope."),
        ("Submission gap", "Required or expected information that is missing or weak in the current submission."),
        ("Traffic-light status", "Green, Amber, Red or Grey rating used to summarize review status."),
        ("Traceability", "Ability to link a finding back to the exact source document, page, table or row."),
        ("Utility-scale", "Large project designed to supply electricity at grid scale rather than household scale."),
        ("Workbook", "Spreadsheet model or Excel file containing assumptions, calculations and outputs."),
    ]
    return [{"term": term, "definition": definition} for term, definition in terms]


def build_report_quality_status(report):
    issues = []
    sensitivity_review = report.get("sensitivity_review", {})
    if sensitivity_review.get("validation_failures"):
        issues.extend(sensitivity_review.get("validation_failures", []))
    financial = report.get("financial_reconciliation", {})
    for row in financial.get("reconciliation_table", []):
        metric_key = row.get("metric_key")
        values_by_source = row.get("values_by_source", {})
        if metric_key == "total_project_cost_usd_mn":
            workbook_values = values_by_source.get("workbook", [])
            unexpected = [value for value in workbook_values if value not in [289, 289.0]]
            if unexpected:
                issues.append(f"Unexpected workbook total project cost values: {unexpected}")
        if metric_key == "minimum_dscr_x":
            workbook_values = values_by_source.get("workbook", [])
            unexpected = [value for value in workbook_values if abs(float(value) - 1.32) > 0.001]
            if unexpected:
                issues.append(f"Unexpected workbook minimum DSCR values: {unexpected}")
    return {"schema_ready": True, "content_ready": len(issues) == 0, "quality_issues": issues}


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
            f"{row.get('preferred_value_for_review')} {row.get('unit')} ({row.get('preferred_source')}) | "
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
    lines.append("| Risk | Rating | Mitigation/status | Reviewer challenge | Status | Sources |")
    lines.append("|---|---|---|---|---|---|")
    for row in risk.get("risk_register", []):
        lines.append(
            f"| {row.get('risk')} | {row.get('rating')} | {clean_text(row.get('mitigation_status'))} | "
            f"{row.get('reviewer_challenge')} | {traffic_icon(row.get('traffic_light'))} {row.get('traffic_light')} | "
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
    lines.append("")

    lines.append("## 15. Terms and Definitions")
    lines.append("")
    lines.append("| Term | Simple definition |")
    lines.append("|---|---|")
    for item in report.get("terms_and_definitions", []):
        lines.append(f"| {item.get('term')} | {item.get('definition')} |")
    lines.append("")

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
    terms_and_definitions = build_terms_and_definitions()

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
        "terms_and_definitions": terms_and_definitions,
    }

    report["report_quality_status"] = build_report_quality_status(report)

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
        "report_quality_status": report.get("report_quality_status"),
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

    print(json.dumps(output, indent=2, ensure_ascii=False))