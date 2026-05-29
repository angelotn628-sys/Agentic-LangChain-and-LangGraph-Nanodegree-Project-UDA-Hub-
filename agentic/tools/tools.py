import json
import sqlite3
from pathlib import Path
import uuid
from datetime import datetime
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("cultpass-support")

BASE_DIR = Path(__file__).resolve().parents[2]
CULTPASS_DB = BASE_DIR / "data" / "external" / "cultpass.db"
UDAHUB_DB = BASE_DIR / "data" / "core" / "udahub.db"

@mcp.tool()
def get_customer_profile(user_id: str) -> dict:
    """Return CultPass customer profile details for a given customer ID."""
    with sqlite3.connect(CULTPASS_DB) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, full_name, email, is_blocked, created_at, updated_at
            FROM users
            WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()

    if not row:
        return {
            "found": False,
            "error": "Customer not found",
            "user_id": user_id
        }

    user = dict(row)

    return {
        "found": True,
        "user": {
            "user_id": user["user_id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "is_blocked": bool(user["is_blocked"]),
            "created_at": user["created_at"],
            "updated_at": user["updated_at"],
        }
    }


@mcp.tool()
def get_customer_reservations(user_id: str) -> dict:
    """Return reservations for a given customer ID."""
    with sqlite3.connect(CULTPASS_DB) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT reservation_id, user_id, experience_id, status, created_at, updated_at
            FROM reservations
            WHERE user_id = ?
        """, (user_id,))
        rows = cursor.fetchall()

    reservations = [dict(row) for row in rows]

    return {
        "found": len(reservations) > 0,
        "user_id": user_id,
        "reservations": reservations
    }


@mcp.tool()
def search_support_articles(query: str) -> dict:
    """Search support articles using lightweight intent-aware keyword matching with ranking."""
    with sqlite3.connect(str(UDAHUB_DB)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query_lower = query.lower().strip()

        primary_keywords = []
        secondary_keywords = []

        if any(word in query_lower for word in ["refund", "charged", "charge", "billing", "payment", "invoice"]):
            primary_keywords = ["refund", "billing", "payment", "charge"]
            secondary_keywords = ["invoice"]

        elif any(word in query_lower for word in ["cancel", "cancellation", "cancelation"]):
            primary_keywords = ["cancel", "cancellation", "cancelation"]
            secondary_keywords = ["subscription", "reservation", "booking"]

        elif any(
            phrase in query_lower
            for phrase in [
                "reschedule",
                "rescheduling",
                "change booking",
                "change reservation",
                "move booking",
                "move reservation",
                "modify booking",
                "modify reservation",
                "edit booking",
                "edit reservation",
            ]
        ):
            primary_keywords = ["reschedule", "change", "modify", "edit"]
            secondary_keywords = ["reservation", "booking"]

        elif any(word in query_lower for word in ["reservation", "booking"]):
            primary_keywords = ["reservation", "booking"]
            secondary_keywords = []

        elif any(word in query_lower for word in ["login", "password", "sign in", "signin", "locked out", "account access"]):
            primary_keywords = ["login", "password", "account"]
            secondary_keywords = []

        else:
            tokens = [word for word in query_lower.split() if len(word) > 2]
            primary_keywords = tokens[:3]
            secondary_keywords = []

        all_keywords = primary_keywords + secondary_keywords

        if not all_keywords:
            return {
                "found": False,
                "query": query,
                "articles": [],
            }

        where_clauses = []
        where_params = []

        score_parts = []
        score_params = []

        for kw in primary_keywords:
            like = f"%{kw}%"
            where_clauses.append(
                "(lower(title) LIKE ? OR lower(content) LIKE ? OR lower(coalesce(tags, '')) LIKE ?)"
            )
            where_params.extend([like, like, like])

            score_parts.append("""
                CASE WHEN lower(title) LIKE ? THEN 12 ELSE 0 END +
                CASE WHEN lower(coalesce(tags, '')) LIKE ? THEN 8 ELSE 0 END +
                CASE WHEN lower(content) LIKE ? THEN 4 ELSE 0 END
            """)
            score_params.extend([like, like, like])

        for kw in secondary_keywords:
            like = f"%{kw}%"
            where_clauses.append(
                "(lower(title) LIKE ? OR lower(content) LIKE ? OR lower(coalesce(tags, '')) LIKE ?)"
            )
            where_params.extend([like, like, like])

            score_parts.append("""
                CASE WHEN lower(title) LIKE ? THEN 5 ELSE 0 END +
                CASE WHEN lower(coalesce(tags, '')) LIKE ? THEN 3 ELSE 0 END +
                CASE WHEN lower(content) LIKE ? THEN 1 ELSE 0 END
            """)
            score_params.extend([like, like, like])

        where_clause = " OR ".join(where_clauses)
        score_expr = " + ".join(f"({part})" for part in score_parts) if score_parts else "0"

        sql = f"""
            SELECT
                article_id,
                account_id,
                title,
                content,
                tags,
                created_at,
                updated_at,
                ({score_expr}) AS relevance_score
            FROM knowledge
            WHERE {where_clause}
            ORDER BY relevance_score DESC, updated_at DESC
            LIMIT 5
        """

        cursor.execute(sql, score_params + where_params)
        rows = cursor.fetchall()

    articles = [dict(row) for row in rows]

    return {
        "found": bool(articles),
        "query": query,
        "articles": articles,
    }



@mcp.tool()
def search_customer_tickets(user_id: str) -> dict:
    """Return support tickets for a given external customer ID."""
    with sqlite3.connect(UDAHUB_DB) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                t.ticket_id,
                t.account_id,
                u.external_user_id AS user_id,
                u.user_name,
                t.channel,
                t.created_at,
                tm.status,
                tm.main_issue_type,
                tm.tags,
                tm.updated_at
            FROM tickets t
            JOIN users u
                ON t.user_id = u.user_id
            LEFT JOIN ticket_metadata tm
                ON t.ticket_id = tm.ticket_id
            WHERE u.external_user_id = ?
            ORDER BY t.created_at DESC
        """, (user_id,))

        rows = cursor.fetchall()

    tickets = [dict(row) for row in rows]

    return {
        "found": len(tickets) > 0,
        "user_id": user_id,
        "tickets": tickets
    }


@mcp.tool()
def create_escalation(
    user_id: str,
    issue_summary: str,
    issue_type: str = "escalation",
    tags: list[str] | None = None,
) -> dict:
    tags = tags or ["human_support"]

    with sqlite3.connect(UDAHUB_DB) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, account_id, external_user_id, user_name
            FROM users
            WHERE external_user_id = ?
        """, (user_id,))
        user_row = cursor.fetchone()

        if not user_row:
            return {
                "created": False,
                "error": "Customer not found in support system",
                "user_id": user_id,
            }

        internal_user_id = user_row["user_id"]
        account_id = user_row["account_id"]

        ticket_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO tickets (ticket_id, account_id, user_id, channel, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (ticket_id, account_id, internal_user_id, "escalation", now))

        cursor.execute("""
            INSERT INTO ticket_metadata (
                ticket_id, status, main_issue_type, tags, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            ticket_id,
            "open",
            issue_type,
            ",".join(tags),
            now,
            now,
        ))

        cursor.execute("""
            INSERT INTO ticket_messages (
                message_id, ticket_id, role, content, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            message_id,
            ticket_id,
            "user",
            issue_summary,
            now,
        ))

        conn.commit()

    return {
        "created": True,
        "ticket": {
            "ticket_id": ticket_id,
            "account_id": account_id,
            "user_id": user_id,
            "channel": "escalation",
            "created_at": now,
            "status": "open",
            "main_issue_type": issue_type,
            "tags": tags,
            "issue_summary": issue_summary,
        },
    }


if __name__ == "__main__":

    print(json.dumps(search_support_articles("how do I cancel a reservation"), indent=2))
    print(json.dumps(search_support_articles("how do I change my reservation"), indent=2))

    mcp.run()