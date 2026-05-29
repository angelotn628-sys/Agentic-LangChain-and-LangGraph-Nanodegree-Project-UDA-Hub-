from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages

from agentic.tools.tools import (
    create_escalation,
    get_customer_profile,
    get_customer_reservations,
    search_customer_tickets,
    search_support_articles,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

ACCOUNT_SPECIFIC_TYPES = {
    "account_access",
    "billing_issue",
    "blocked_account",
}

AUTO_RESOLVABLE_TYPES = {
    "faq",
    "policy_question",
    "general_info",
    "reservation_issue",
}

HUMAN_REQUIRED_TYPES = {
    "safety_incident",
    "blocked_account",
    "billing_issue",
    "account_access",
}

OPEN_ESCALATION_STATUSES = {"open", "pending"}

IssueType = Literal[
    "faq",
    "policy_question",
    "how_to",
    "general_info",
    "account_access",
    "reservation_issue",
    "billing_issue",
    "billing_dispute",
    "blocked_account",
    "reservation_dispute",
    "safety_incident",
    "other",
]

ResolverOutcome = Literal[
    "resolved",
    "unresolved",
    "needs_escalation",
]

ResolutionConfidence = Literal[
    "high",
    "medium",
    "low",
    "unknown",
]


class SupportState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    user_id: str
    user_message: str
    issue_type: IssueType
    issue_summary: str

    needs_escalation: bool
    create_new_escalation: bool
    needs_human_review: bool

    final_response: str
    error: str

    profile: dict[str, Any] | None
    customer_profile: dict[str, Any] | None
    reservations: dict[str, Any]
    tickets: dict[str, Any]
    articles: dict[str, Any]

    investigation_summary: str
    customer_context_summary: str
    kb_query: str
    kb_summary: str

    resolver_outcome: ResolverOutcome
    resolution_confidence: ResolutionConfidence
    safe_to_auto_resolve: bool
    requires_customer_context: bool
    customer_message: str

    escalation_reason: str
    escalation_tags: list[str]
    escalation_result: dict[str, Any]

    existing_escalation_status: str
    routing: dict[str, Any]


class EscalationDecision(TypedDict):
    needs_human_review: bool
    create_new_escalation: bool
    reason: str
    tags: list[str]


def has_open_escalation(tickets: dict[str, Any]) -> bool:
    for ticket in tickets.get("tickets", []):
        status = str(ticket.get("status", "")).strip().lower()
        main_issue_type = str(ticket.get("main_issue_type", "")).strip().lower()
        channel = str(ticket.get("channel", "")).strip().lower()

        raw_tags = ticket.get("tags", [])
        if isinstance(raw_tags, str):
            tags = {t.strip().lower() for t in raw_tags.split(",") if t.strip()}
        elif isinstance(raw_tags, list):
            tags = {str(t).strip().lower() for t in raw_tags if str(t).strip()}
        else:
            tags = set()

        if (
            status in OPEN_ESCALATION_STATUSES
            and main_issue_type == "escalation"
            and channel == "escalation"
            and "human_support" in tags
        ):
            return True

    return False


def normalize_resolver_outcome(value: Any) -> ResolverOutcome:
    if value is None:
        return "unresolved"

    v = str(value).strip().lower()
    mapping = {
        "resolved": "resolved",
        "answer_provided": "resolved",
        "completed": "resolved",
        "done": "resolved",
        "unresolved": "unresolved",
        "needs_more_info": "unresolved",
        "partial": "unresolved",
        "cannot_resolve": "unresolved",
        "needs_escalation": "needs_escalation",
        "escalate": "needs_escalation",
        "escalated": "needs_escalation",
        "human_handoff": "needs_escalation",
        "handoff": "needs_escalation",
    }
    return mapping.get(v, "unresolved")


def normalize_confidence(value: Any) -> ResolutionConfidence:
    v = str(value).strip().lower()
    if v in {"high", "medium", "low", "unknown"}:
        return v
    return "unknown"


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No valid JSON object found in model response.")


def parse_resolver_response(raw_content: str) -> dict[str, Any]:
    try:
        parsed = extract_json_object(raw_content)
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}

    resolver_outcome = normalize_resolver_outcome(parsed.get("resolver_outcome"))
    resolution_confidence = normalize_confidence(parsed.get("resolution_confidence"))

    safe_to_auto_resolve = parsed.get("safe_to_auto_resolve", False)
    if not isinstance(safe_to_auto_resolve, bool):
        if isinstance(safe_to_auto_resolve, str):
            safe_to_auto_resolve = safe_to_auto_resolve.strip().lower() == "true"
        else:
            safe_to_auto_resolve = False

    requires_customer_context = parsed.get("requires_customer_context", False)
    if not isinstance(requires_customer_context, bool):
        if isinstance(requires_customer_context, str):
            requires_customer_context = (
                requires_customer_context.strip().lower() == "true"
            )
        else:
            requires_customer_context = False

    customer_message = str(
        parsed.get(
            "customer_message",
            "I’m not fully confident I can resolve this automatically.",
        )
    ).strip()

    if not customer_message:
        customer_message = "I’m not fully confident I can resolve this automatically."

    return {
        "resolver_outcome": resolver_outcome,
        "resolution_confidence": resolution_confidence,
        "safe_to_auto_resolve": safe_to_auto_resolve,
        "requires_customer_context": requires_customer_context,
        "customer_message": customer_message,
    }


def _latest_user_text(state: SupportState) -> str:
    messages = state.get("messages", []) or []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return (state.get("user_message") or "").strip()


def classify_issue_node(state: SupportState) -> SupportState:
    user_message = _latest_user_text(state)
    message = user_message.lower().strip()

    safety_keywords = [
        "unsafe",
        "dangerous",
        "threat",
        "threatened",
        "harassed",
        "harassment",
        "assault",
        "injured",
        "injury",
        "emergency",
        "fire",
        "violence",
        "smoke",
        "gas leak",
        "break in",
        "break-in",
    ]

    blocked_account_keywords = [
        "account locked",
        "my account is locked",
        "locked out",
        "blocked account",
        "account blocked",
        "my account is blocked",
        "cannot log in",
        "can't log in",
        "cant log in",
        "cannot login",
        "can't login",
        "cant login",
        "unable to log in",
        "unable to login",
        "cannot access my account",
        "can't access my account",
        "cant access my account",
        "unable to access my account",
        "blocked from my account",
        "login blocked",
        "access denied",
        "password reset not working",
        "2fa issue",
        "verification failed",
    ]

    billing_keywords = [
        "refund",
        "charged",
        "charge",
        "billing",
        "payment",
        "invoice",
        "receipt",
        "double charge",
        "charged twice",
        "charged me twice",
        "duplicate charge",
        "extra charge",
        "unknown charge",
        "unknown charges",
        "charge i don't recognize",
        "charge i do not recognize",
        "mystery charge",
        "unexpected charge",
        "unauthorized charge",
        "unauthorized billing",
        "chargeback",
        "fraud",
        "no-show fee",
    ]

    reservation_keywords = [
        "reservation",
        "booking",
        "booked",
        "cancel my booking",
        "cancel reservation",
        "change my booking",
        "modify booking",
        "edit booking",
        "reschedule",
        "move my reservation",
        "check in",
        "check-in",
        "check out",
        "check-out",
        "confirmation number",
    ]

    policy_keywords = [
        "policy",
        "rules",
        "terms",
        "allowed",
        "pet policy",
        "cancellation policy",
        "refund policy",
        "house rules",
    ]

    faq_keywords = [
        "wifi",
        "parking",
        "pool",
        "hours",
        "address",
        "location",
        "amenities",
        "check-in time",
        "checkout time",
    ]

    general_info_keywords = [
        "tell me about",
        "what is",
        "how does",
        "information about",
    ]

    blocked_account_signals = [
        "locked",
        "blocked",
        "login",
        "log in",
        "access",
        "password",
        "2fa",
        "verification",
    ]
    account_signals = ["account", "profile"]

    if any(keyword in message for keyword in safety_keywords):
        issue_type: IssueType = "safety_incident"
    elif any(keyword in message for keyword in billing_keywords):
        issue_type = "billing_issue"
    elif (
        any(keyword in message for keyword in blocked_account_keywords)
        or (
            any(signal in message for signal in blocked_account_signals)
            and any(signal in message for signal in account_signals)
        )
    ):
        issue_type = "blocked_account"
    elif any(keyword in message for keyword in reservation_keywords):
        issue_type = "reservation_issue"
    elif any(keyword in message for keyword in policy_keywords):
        issue_type = "policy_question"
    elif any(keyword in message for keyword in faq_keywords):
        issue_type = "faq"
    elif any(keyword in message for keyword in general_info_keywords):
        issue_type = "general_info"
    else:
        issue_type = "other"

    return {
        **state,
        "user_message": user_message,
        "issue_type": issue_type,
    }


def customer_context_node(state: SupportState) -> SupportState:
    user_id = state.get("user_id")

    if not user_id:
        return {**state, "error": "Missing user_id."}

    try:
        profile = get_customer_profile(user_id)
        reservations = get_customer_reservations(user_id)
        tickets = search_customer_tickets(user_id)

        existing_escalation_status = ""
        if isinstance(tickets, dict):
            for ticket in tickets.get("tickets", []):
                status = str(ticket.get("status", "")).strip().lower()
                if status in OPEN_ESCALATION_STATUSES:
                    existing_escalation_status = status
                    break

        return {
            **state,
            "customer_profile": profile,
            "profile": profile,
            "reservations": reservations,
            "tickets": tickets,
            "existing_escalation_status": existing_escalation_status,
            "error": "",
        }
    except Exception as e:
        return {
            **state,
            "error": f"Customer context lookup failed: {str(e)}",
        }


def kb_retrieval_node(state: SupportState) -> SupportState:
    user_message = _latest_user_text(state)
    kb_query = (state.get("kb_query") or "").strip()
    issue_summary = (state.get("issue_summary") or "").strip()

    query = kb_query or user_message.strip() or issue_summary

    if not query:
        return {
            **state,
            "articles": {"found": False, "query": "", "articles": []},
            "kb_query": "",
            "error": "Missing KB query.",
        }

    try:
        articles = search_support_articles(query)
        return {
            **state,
            "articles": articles,
            "kb_query": query,
            "error": "",
        }
    except Exception as e:
        return {
            **state,
            "articles": {"found": False, "query": query, "articles": []},
            "kb_query": query,
            "error": f"KB retrieval failed: {str(e)}",
        }


def should_escalate(state: SupportState) -> EscalationDecision:
    issue_type = state.get("issue_type", "other")
    customer_profile = state.get("customer_profile")
    resolver_outcome = state.get("resolver_outcome", "unresolved")
    safe_to_auto_resolve = state.get("safe_to_auto_resolve", False)
    requires_customer_context = state.get("requires_customer_context", False)
    existing_status = str(state.get("existing_escalation_status", "")).strip().lower()

    if existing_status in OPEN_ESCALATION_STATUSES:
        return {
            "needs_human_review": True,
            "create_new_escalation": False,
            "reason": "existing_open_escalation",
            "tags": ["human_support", "existing_open_escalation"],
        }

    if customer_profile is None and issue_type in {
        "blocked_account",
        "billing_issue",
    }:
        return {
            "needs_human_review": True,
            "create_new_escalation": True,
            "reason": "missing_customer_profile",
            "tags": ["human_support", "missing_customer_profile", issue_type],
        }

    if (
        customer_profile is None
        and issue_type == "reservation_issue"
        and requires_customer_context
    ):
        return {
            "needs_human_review": True,
            "create_new_escalation": True,
            "reason": "missing_customer_profile",
            "tags": ["human_support", "missing_customer_profile", issue_type],
        }

    if issue_type in {"safety_incident", "blocked_account", "billing_issue"}:
        if resolver_outcome == "resolved" and safe_to_auto_resolve:
            return {
                "needs_human_review": False,
                "create_new_escalation": False,
                "reason": "resolved_safely",
                "tags": [],
            }
        return {
            "needs_human_review": True,
            "create_new_escalation": True,
            "reason": "human_required_issue",
            "tags": ["human_support", issue_type],
        }

    if issue_type == "other" and resolver_outcome in {
        "needs_escalation",
        "unresolved",
    }:
        return {
            "needs_human_review": True,
            "create_new_escalation": True,
            "reason": "other_needs_escalation",
            "tags": ["human_support", "other"],
        }

    return {
        "needs_human_review": False,
        "create_new_escalation": False,
        "reason": "auto_handle",
        "tags": [],
    }


def resolver_node(state: SupportState) -> SupportState:
    user_message = _latest_user_text(state)
    customer_profile = state.get("customer_profile") or {}
    reservations = state.get("reservations") or {}
    tickets = state.get("tickets") or {}
    articles = state.get("articles") or {}

    user = customer_profile.get("user") or {}

    reservation_list = (
        reservations.get("reservations", []) if isinstance(reservations, dict) else []
    )
    ticket_list = tickets.get("tickets", []) if isinstance(tickets, dict) else []
    article_list = articles.get("articles", []) if isinstance(articles, dict) else []

    article_found = bool(article_list)
    general_guidance_possible = (
        bool(article_list)
        and state.get("issue_type") in {
            "reservation_issue",
            "policy_question",
            "faq",
            "general_info",
        }
    )

    grounded_context = {
        "issue_type": state.get("issue_type", ""),
        "user_message": user_message,
        "issue_summary": state.get("issue_summary", ""),
        "profile_found": customer_profile.get("found", False),
        "user_blocked": user.get("is_blocked", False),
        "reservation_count": len(reservation_list),
        "reservations_preview": reservation_list[:3],
        "open_ticket_count": len(ticket_list),
        "ticket_statuses": [
            t.get("status") for t in ticket_list[:5] if isinstance(t, dict)
        ],
        "article_found": article_found,
        "article_count": len(article_list),
        "articles_preview": [
            {
                "title": a.get("title", ""),
                "tags": a.get("tags", ""),
                "content_preview": (
                    a.get("content", "")[:300] if a.get("content") else ""
                ),
            }
            for a in article_list[:3]
            if isinstance(a, dict)
        ],
        "general_guidance_possible": general_guidance_possible,
    }

    prompt = f"""
You are a customer support resolver.

Decide whether this issue is:
- resolved
- unresolved
- needs_escalation

You must be conservative about escalation.

Important guidance:
- General guidance, FAQ, policy explanations, cancellation steps, rescheduling steps, and booking how-to questions should usually NOT be escalated.
- If the available articles or customer context support a helpful next-step answer, mark the issue as "resolved".
- "Resolved" does NOT require completing a backend action. It is enough to give safe, grounded guidance based on the provided context.
- Use "unresolved" only when the available information is not enough to answer safely, and human intervention is not clearly required.
- Use "needs_escalation" only when the issue clearly requires human staff action, sensitive intervention, account intervention, dispute handling, safety handling, or manual review.
- Do not invent policies, refunds, credits, reservation changes, or account actions.

Reservation-specific guidance:
- Do NOT require reservation details for generic how-to questions such as:
  - how to cancel a reservation
  - how to reschedule or change a booking
  - general booking help
- If support articles provide relevant guidance for those questions, mark the issue as "resolved".
- Require customer context when the user is asking about a specific existing reservation or requesting action on a specific booking.
- Treat messages about "my booking", "my reservation", "my trip", "this reservation", or requests to cancel/change/reschedule an existing booking as customer-context-dependent.
- Generic questions like "how do I cancel a reservation?" are NOT customer-context-dependent.

Grounded context:
{grounded_context}

Return ONLY valid JSON with exactly these keys:
{{
  "resolver_outcome": "resolved" | "unresolved" | "needs_escalation",
  "resolution_confidence": "high" | "medium" | "low" | "unknown",
  "safe_to_auto_resolve": true | false,
  "requires_customer_context": true | false,
  "customer_message": "short customer-facing reply"
}}

Decision rules:
- Use "resolved" only if the reply is supported by the provided articles or customer context.
- For reservation_issue, policy_question, faq, and general_info:
  - prefer "resolved" when you can provide grounded guidance or explain next steps
  - use "unresolved" only if the context/articles are too weak to answer safely
- For blocked_account, billing_issue, and safety_incident:
  - use "needs_escalation" when staff review or intervention is clearly needed
- safe_to_auto_resolve should be true only when the customer_message is safe, grounded, and does not promise unsupported actions.
- requires_customer_context should be true when answering safely depends on identifying the user's account, reservation, or a specific existing booking/request.
- For generic how-to or policy questions, requires_customer_context should be false.
- customer_message should directly answer the user's question when possible, using the grounded article/context information.
- customer_message must be short, specific, and must not claim actions were taken unless the context explicitly shows that.
"""

    try:
        result = llm.invoke(prompt)
        raw_content = (
            result.content if isinstance(result.content, str) else str(result.content)
        )

        parsed = parse_resolver_response(raw_content)

        return {
            **state,
            "resolver_outcome": parsed["resolver_outcome"],
            "resolution_confidence": parsed["resolution_confidence"],
            "safe_to_auto_resolve": parsed["safe_to_auto_resolve"],
            "requires_customer_context": parsed["requires_customer_context"],
            "customer_message": parsed["customer_message"],
            "error": "",
        }
    except Exception as e:
        return {
            **state,
            "resolver_outcome": "unresolved",
            "resolution_confidence": "unknown",
            "safe_to_auto_resolve": False,
            "requires_customer_context": False,
            "customer_message": "I wasn’t able to confidently resolve this automatically.",
            "error": f"Resolver failed: {str(e)}",
        }


def escalation_decider_node(state: SupportState) -> SupportState:
    decision = should_escalate(state)

    return {
        **state,
        "routing": {
            "needs_human_review": decision["needs_human_review"],
            "needs_escalation": decision["needs_human_review"],
            "create_new_escalation": decision["create_new_escalation"],
            "reason": decision["reason"],
        },
        "needs_human_review": decision["needs_human_review"],
        "needs_escalation": decision["needs_human_review"],
        "create_new_escalation": decision["create_new_escalation"],
        "escalation_reason": decision["reason"],
        "escalation_tags": decision["tags"],
    }


def create_escalation_node(state: SupportState) -> SupportState:
    user_id = state.get("user_id")
    issue_summary = (
        state.get("issue_summary")
        or _latest_user_text(state)
        or "Customer needs support"
    )
    issue_type = state.get("issue_type", "other")
    tags = state.get("escalation_tags", [])

    if not user_id:
        return {
            **state,
            "escalation_result": {
                "created": False,
                "error": "Missing user_id for escalation creation.",
            },
            "error": "Escalation required but user_id is missing.",
        }

    result = create_escalation(
        user_id=user_id,
        issue_summary=issue_summary,
        issue_type=issue_type,
        tags=tags,
    )

    return {
        **state,
        "escalation_result": result,
    }


def respond_node(state: SupportState) -> SupportState:
    if state.get("needs_escalation"):
        if state.get("create_new_escalation"):
            final_response = (
                "I wasn’t able to safely resolve this automatically, so I created a support escalation for human review."
            )
        else:
            final_response = (
                "This issue already has an open escalation, so it will continue with human support."
            )
    else:
        final_response = (
            state.get("customer_message") or "Here’s the best answer I could find."
        )

    return {
        **state,
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


def route_customer_context(state: SupportState) -> str:
    if state.get("issue_type") in ACCOUNT_SPECIFIC_TYPES:
        return "retrieve_customer_context"
    return "retrieve_kb"


def route_escalation_creation(state: SupportState) -> str:
    if state.get("create_new_escalation"):
        return "create_escalation"
    return "respond"





