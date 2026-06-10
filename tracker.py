"""
╔══════════════════════════════════════════════════════════╗
║         NEXT LEVEL PERSONAL FINANCE TRACKER              ║
║  Features: Transactions · Budgets · Charts · Reports     ║
║  Stack: SQLite (storage) · Rich (TUI) · Plotext (charts) ║
╚══════════════════════════════════════════════════════════╝

SETUP (one-time):
    pip install rich plotext

RUN:
    python finance_tracker.py
"""

import sqlite3
import os
import sys
from datetime import datetime, date
from typing import Optional

# ── Third-party ──────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich import box
    from rich.text import Text
    from rich.columns import Columns
    from rich.rule import Rule
    import plotext as plt
except ImportError:
    print("Missing deps! Run:  pip install rich plotext")
    sys.exit(1)

# ── Globals ───────────────────────────────────────────────
console = Console()
DB_PATH = "finance.db"  # SQLite file lives next to the script

# ── Emoji + colour maps for categories ───────────────────
CATEGORY_STYLE = {
    "Food":          ("🍔", "yellow"),
    "Transport":     ("🚗", "blue"),
    "Entertainment": ("🎬", "magenta"),
    "Health":        ("💊", "green"),
    "Shopping":      ("🛍️",  "cyan"),
    "Utilities":     ("💡", "bright_yellow"),
    "Rent":          ("🏠", "red"),
    "Salary":        ("💰", "bright_green"),
    "Freelance":     ("💼", "bright_cyan"),
    "Investment":    ("📈", "bright_magenta"),
    "Other":         ("📦", "white"),
}

CATEGORIES = list(CATEGORY_STYLE.keys())

# ═══════════════════════════════════════════════════════════
#  DATABASE LAYER
#  All SQLite logic lives here.  The rest of the app uses
#  these functions and never touches raw SQL directly.
# ═══════════════════════════════════════════════════════════

def init_db() -> sqlite3.Connection:
    """
    Create (or open) the SQLite database and ensure the
    tables exist.  Returns an open connection.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    cur = conn.cursor()

    # Transactions table  ────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            date      TEXT    NOT NULL,          -- ISO-8601 YYYY-MM-DD
            type      TEXT    NOT NULL,          -- 'income' | 'expense'
            amount    REAL    NOT NULL,
            category  TEXT    NOT NULL,
            note      TEXT    DEFAULT ''
        )
    """)

    # Monthly budgets table  ─────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            month     TEXT    NOT NULL,          -- YYYY-MM
            category  TEXT    NOT NULL,
            limit_amt REAL    NOT NULL,
            UNIQUE(month, category)              -- one budget per cat/month
        )
    """)

    conn.commit()
    return conn


def add_transaction(conn, date_str, t_type, amount, category, note=""):
    """Insert one transaction row."""
    conn.execute(
        "INSERT INTO transactions (date, type, amount, category, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (date_str, t_type, amount, category, note)
    )
    conn.commit()


def delete_transaction(conn, tx_id: int) -> bool:
    """Delete by ID.  Returns True if a row was actually deleted."""
    cur = conn.execute("DELETE FROM transactions WHERE id = ?", (tx_id,))
    conn.commit()
    return cur.rowcount > 0


def get_transactions(conn, month: Optional[str] = None,
                     t_type: Optional[str] = None) -> list:
    """
    Fetch transactions, optionally filtered by month (YYYY-MM)
    and/or type ('income' | 'expense').
    """
    query = "SELECT * FROM transactions WHERE 1=1"
    params = []
    if month:
        query += " AND strftime('%Y-%m', date) = ?"
        params.append(month)
    if t_type:
        query += " AND type = ?"
        params.append(t_type)
    query += " ORDER BY date DESC, id DESC"
    return conn.execute(query, params).fetchall()


def get_monthly_summary(conn, month: str) -> dict:
    """
    Return {category: total_spent} for expense transactions
    in the given month (YYYY-MM).
    """
    rows = conn.execute("""
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE strftime('%Y-%m', date) = ?
          AND type = 'expense'
        GROUP BY category
    """, (month,)).fetchall()
    return {r["category"]: r["total"] for r in rows}


def upsert_budget(conn, month: str, category: str, limit_amt: float):
    """Insert or update a budget limit for a category/month."""
    conn.execute("""
        INSERT INTO budgets (month, category, limit_amt)
        VALUES (?, ?, ?)
        ON CONFLICT(month, category) DO UPDATE SET limit_amt = excluded.limit_amt
    """, (month, category, limit_amt))
    conn.commit()


def get_budgets(conn, month: str) -> dict:
    """Return {category: limit_amt} for a given month."""
    rows = conn.execute(
        "SELECT category, limit_amt FROM budgets WHERE month = ?", (month,)
    ).fetchall()
    return {r["category"]: r["limit_amt"] for r in rows}


def get_all_months(conn) -> list:
    """Return a sorted list of all months (YYYY-MM) that have transactions."""
    rows = conn.execute("""
        SELECT DISTINCT strftime('%Y-%m', date) as m
        FROM transactions
        ORDER BY m DESC
    """).fetchall()
    return [r["m"] for r in rows]


# ═══════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════

def clear():
    """Cross-platform terminal clear."""
    os.system("cls" if os.name == "nt" else "clear")


def current_month() -> str:
    """Today's month as YYYY-MM string."""
    return date.today().strftime("%Y-%m")


def pick_category() -> str:
    """Show a numbered list of categories and return the chosen one."""
    console.print("\n[bold]Categories:[/bold]")
    for i, cat in enumerate(CATEGORIES, 1):
        emoji, colour = CATEGORY_STYLE[cat]
        console.print(f"  [{colour}]{i:>2}. {emoji} {cat}[/{colour}]")
    while True:
        raw = Prompt.ask("Pick a number")
        if raw.isdigit() and 1 <= int(raw) <= len(CATEGORIES):
            return CATEGORIES[int(raw) - 1]
        console.print("[red]Invalid choice, try again.[/red]")


def fmt_amount(amount: float, t_type: str) -> Text:
    """Colour-code amounts: green for income, red for expense."""
    sign = "+" if t_type == "income" else "-"
    colour = "bright_green" if t_type == "income" else "bright_red"
    return Text(f"{sign}₹{amount:,.2f}", style=colour)


def print_header(title: str):
    """Print a decorative section header."""
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()


# ═══════════════════════════════════════════════════════════
#  FEATURE: ADD TRANSACTION
# ═══════════════════════════════════════════════════════════

def add_transaction_flow(conn):
    """Interactive wizard to record a new income or expense."""
    print_header("Add Transaction")

    # ── Type ─────────────────────────────────────────────
    t_type = Prompt.ask("Type", choices=["income", "expense"], default="expense")

    # ── Amount ───────────────────────────────────────────
    while True:
        raw = Prompt.ask("Amount (₹)")
        try:
            amount = float(raw)
            if amount <= 0:
                raise ValueError
            break
        except ValueError:
            console.print("[red]Enter a positive number.[/red]")

    # ── Category ─────────────────────────────────────────
    category = pick_category()

    # ── Date (default today) ─────────────────────────────
    today_str = date.today().isoformat()
    date_str = Prompt.ask("Date (YYYY-MM-DD)", default=today_str)
    try:
        datetime.strptime(date_str, "%Y-%m-%d")   # validate format
    except ValueError:
        console.print("[yellow]Bad date format, using today.[/yellow]")
        date_str = today_str

    # ── Optional note ────────────────────────────────────
    note = Prompt.ask("Note (optional)", default="")

    add_transaction(conn, date_str, t_type, amount, category, note)
    console.print(f"\n[bold green]✓ Transaction saved![/bold green]")


# ═══════════════════════════════════════════════════════════
#  FEATURE: VIEW TRANSACTIONS
# ═══════════════════════════════════════════════════════════

def view_transactions(conn):
    """Display a paginated table of transactions for a chosen month."""
    print_header("Transactions")

    months = get_all_months(conn)
    month = Prompt.ask("Month (YYYY-MM)", default=current_month())

    rows = get_transactions(conn, month=month)

    if not rows:
        console.print("[yellow]No transactions found for this month.[/yellow]")
        return

    # ── Build Rich table ─────────────────────────────────
    table = Table(
        title=f"Transactions — {month}",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("ID",       style="dim",          width=5)
    table.add_column("Date",                           width=12)
    table.add_column("Type",                           width=9)
    table.add_column("Category",                       width=16)
    table.add_column("Amount",   justify="right",      width=14)
    table.add_column("Note",     style="italic dim",   width=20)

    total_income = total_expense = 0.0

    for r in rows:
        emoji, colour = CATEGORY_STYLE.get(r["category"], ("📦", "white"))
        cat_label = f"[{colour}]{emoji} {r['category']}[/{colour}]"

        if r["type"] == "income":
            total_income += r["amount"]
        else:
            total_expense += r["amount"]

        table.add_row(
            str(r["id"]),
            r["date"],
            f"[{'green' if r['type']=='income' else 'red'}]{r['type']}[/]",
            cat_label,
            fmt_amount(r["amount"], r["type"]),
            r["note"] or "—",
        )

    console.print(table)

    # ── Footer summary ───────────────────────────────────
    net = total_income - total_expense
    net_colour = "green" if net >= 0 else "red"
    console.print(
        f"\n  [bright_green]Income : ₹{total_income:>12,.2f}[/bright_green]  "
        f"[bright_red]Expenses: ₹{total_expense:>12,.2f}[/bright_red]  "
        f"[{net_colour}]Net: ₹{net:>12,.2f}[/{net_colour}]"
    )


# ═══════════════════════════════════════════════════════════
#  FEATURE: DELETE TRANSACTION
# ═══════════════════════════════════════════════════════════

def delete_transaction_flow(conn):
    """Ask for a transaction ID and confirm before deleting."""
    print_header("Delete Transaction")
    raw = Prompt.ask("Enter transaction ID to delete")
    if not raw.isdigit():
        console.print("[red]Invalid ID.[/red]")
        return
    tx_id = int(raw)
    if Confirm.ask(f"Delete transaction #{tx_id}?"):
        if delete_transaction(conn, tx_id):
            console.print("[green]Deleted.[/green]")
        else:
            console.print("[yellow]No transaction with that ID.[/yellow]")


# ═══════════════════════════════════════════════════════════
#  FEATURE: BUDGET MANAGEMENT
# ═══════════════════════════════════════════════════════════

def manage_budgets(conn):
    """Set monthly spending limits per category, then show status."""
    print_header("Budget Manager")

    month = Prompt.ask("Month (YYYY-MM)", default=current_month())
    existing = get_budgets(conn, month)
    spent    = get_monthly_summary(conn, month)

    console.print(f"\n[bold]Setting budgets for [cyan]{month}[/cyan][/bold]")
    console.print("(Press Enter to keep current limit, 0 to remove)\n")

    # ── Input loop for each category ─────────────────────
    for cat in CATEGORIES:
        emoji, colour = CATEGORY_STYLE[cat]
        current = existing.get(cat, 0)
        hint = f"₹{current:,.0f}" if current else "none"
        raw = Prompt.ask(
            f"  [{colour}]{emoji} {cat}[/{colour}] limit [{hint}]",
            default=""
        )
        if raw == "":
            continue                      # keep existing
        try:
            val = float(raw)
            if val == 0:
                conn.execute(
                    "DELETE FROM budgets WHERE month=? AND category=?",
                    (month, cat)
                )
                conn.commit()
            else:
                upsert_budget(conn, month, cat, val)
        except ValueError:
            console.print(f"    [red]Skipped (bad value)[/red]")

    # ── Show budget status table ──────────────────────────
    budgets = get_budgets(conn, month)
    if not budgets:
        console.print("\n[yellow]No budgets set for this month.[/yellow]")
        return

    table = Table(
        title=f"Budget Status — {month}",
        box=box.SIMPLE_HEAVY,
        header_style="bold white",
    )
    table.add_column("Category",  width=18)
    table.add_column("Budget",    justify="right", width=12)
    table.add_column("Spent",     justify="right", width=12)
    table.add_column("Remaining", justify="right", width=12)
    table.add_column("Progress",  width=24)

    for cat, limit in sorted(budgets.items()):
        sp = spent.get(cat, 0)
        remaining = limit - sp
        pct = min(sp / limit, 1.0) if limit > 0 else 0

        # ASCII progress bar (20 chars wide)
        filled = int(pct * 20)
        bar_colour = "green" if pct < 0.75 else ("yellow" if pct < 1.0 else "red")
        bar = f"[{bar_colour}]{'█' * filled}{'░' * (20 - filled)}[/{bar_colour}] {pct*100:.0f}%"

        rem_colour = "green" if remaining >= 0 else "red"
        emoji, colour = CATEGORY_STYLE[cat]

        table.add_row(
            f"[{colour}]{emoji} {cat}[/{colour}]",
            f"₹{limit:,.0f}",
            f"₹{sp:,.0f}",
            f"[{rem_colour}]₹{remaining:,.0f}[/{rem_colour}]",
            bar,
        )

    console.print(table)


# ═══════════════════════════════════════════════════════════
#  FEATURE: SPENDING CHART (bar chart via plotext)
# ═══════════════════════════════════════════════════════════

def spending_chart(conn):
    """
    Render a horizontal bar chart of expenses by category
    for a chosen month, drawn directly in the terminal.
    """
    print_header("Spending Chart")

    month = Prompt.ask("Month (YYYY-MM)", default=current_month())
    summary = get_monthly_summary(conn, month)

    if not summary:
        console.print("[yellow]No expense data for this month.[/yellow]")
        return

    # Sort by amount descending for readability
    sorted_data = sorted(summary.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{CATEGORY_STYLE[cat][0]} {cat}" for cat, _ in sorted_data]
    values = [amt for _, amt in sorted_data]

    plt.clear_figure()
    plt.bar(labels, values, orientation="horizontal", color="cyan")
    plt.title(f"Expenses by Category — {month}")
    plt.xlabel("Amount (₹)")
    plt.theme("dark")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  FEATURE: MONTHLY TREND CHART (line chart)
# ═══════════════════════════════════════════════════════════

def trend_chart(conn):
    """
    Plot income vs. expenses for the last N months as a
    line chart in the terminal.
    """
    print_header("Income vs Expenses Trend")

    months = get_all_months(conn)
    if len(months) < 2:
        console.print("[yellow]Need at least 2 months of data for a trend.[/yellow]")
        return

    # ── Aggregate per month ───────────────────────────────
    income_vals, expense_vals, labels = [], [], []
    for m in reversed(months[:12]):   # last 12 months, oldest first
        rows = get_transactions(conn, month=m)
        inc = sum(r["amount"] for r in rows if r["type"] == "income")
        exp = sum(r["amount"] for r in rows if r["type"] == "expense")
        income_vals.append(inc)
        expense_vals.append(exp)
        labels.append(m[5:])   # show "MM" only to save space

    plt.clear_figure()
    plt.plot(labels, income_vals,  label="Income",   color="green")
    plt.plot(labels, expense_vals, label="Expenses", color="red")
    plt.title("Monthly Income vs Expenses (last 12 months)")
    plt.xlabel("Month")
    plt.ylabel("Amount (₹)")
    plt.theme("dark")
    plt.show()


# ═══════════════════════════════════════════════════════════
#  FEATURE: FULL MONTHLY REPORT
# ═══════════════════════════════════════════════════════════

def monthly_report(conn):
    """
    Print a comprehensive text report for a chosen month:
    totals, savings rate, category breakdown, and budget health.
    """
    print_header("Monthly Report")

    month = Prompt.ask("Month (YYYY-MM)", default=current_month())

    rows    = get_transactions(conn, month=month)
    budgets = get_budgets(conn, month)
    spent   = get_monthly_summary(conn, month)

    if not rows:
        console.print("[yellow]No data for this month.[/yellow]")
        return

    total_income  = sum(r["amount"] for r in rows if r["type"] == "income")
    total_expense = sum(r["amount"] for r in rows if r["type"] == "expense")
    net           = total_income - total_expense
    savings_rate  = (net / total_income * 100) if total_income > 0 else 0
    sr_colour     = "green" if savings_rate >= 20 else ("yellow" if savings_rate >= 0 else "red")

    # ── Summary panel ─────────────────────────────────────
    summary_text = (
        f"[bold]Month:[/bold]         [cyan]{month}[/cyan]\n"
        f"[bold]Total Income:[/bold]  [bright_green]₹{total_income:>12,.2f}[/bright_green]\n"
        f"[bold]Total Expenses:[/bold][bright_red]₹{total_expense:>12,.2f}[/bright_red]\n"
        f"[bold]Net Savings:[/bold]   [{'green' if net>=0 else 'red'}]₹{net:>12,.2f}[/]\n"
        f"[bold]Savings Rate:[/bold]  [{sr_colour}]{savings_rate:.1f}%[/{sr_colour}]"
    )
    console.print(Panel(summary_text, title="📊 Summary", border_style="cyan"))

    # ── Category breakdown ────────────────────────────────
    if spent:
        cat_table = Table(box=box.SIMPLE, header_style="bold magenta")
        cat_table.add_column("Category")
        cat_table.add_column("Spent",    justify="right")
        cat_table.add_column("% of Total", justify="right")
        cat_table.add_column("Budget",   justify="right")
        cat_table.add_column("Status")

        for cat, amt in sorted(spent.items(), key=lambda x: -x[1]):
            emoji, colour = CATEGORY_STYLE.get(cat, ("📦", "white"))
            pct = (amt / total_expense * 100) if total_expense else 0
            bud = budgets.get(cat)
            if bud:
                over = amt > bud
                status = "[red]OVER BUDGET[/red]" if over else "[green]Within budget[/green]"
                bud_str = f"₹{bud:,.0f}"
            else:
                status = "[dim]No budget[/dim]"
                bud_str = "—"
            cat_table.add_row(
                f"[{colour}]{emoji} {cat}[/{colour}]",
                f"₹{amt:,.2f}",
                f"{pct:.1f}%",
                bud_str,
                status,
            )

        console.print(Panel(cat_table, title="📂 Category Breakdown", border_style="magenta"))

    # ── Budget alerts ─────────────────────────────────────
    alerts = [
        f"[red]⚠  {cat}: ₹{spent.get(cat, 0):,.0f} spent of ₹{lim:,.0f} budget[/red]"
        for cat, lim in budgets.items()
        if spent.get(cat, 0) > lim
    ]
    if alerts:
        console.print(Panel("\n".join(alerts), title="🚨 Budget Alerts", border_style="red"))


# ═══════════════════════════════════════════════════════════
#  FEATURE: DASHBOARD (quick overview at startup)
# ═══════════════════════════════════════════════════════════

def dashboard(conn):
    """
    Show a compact overview panel for the current month:
    income, expenses, net, top spending category, budget warnings.
    """
    month   = current_month()
    rows    = get_transactions(conn, month=month)
    budgets = get_budgets(conn, month)
    spent   = get_monthly_summary(conn, month)

    income  = sum(r["amount"] for r in rows if r["type"] == "income")
    expense = sum(r["amount"] for r in rows if r["type"] == "expense")
    net     = income - expense
    tx_count = len(rows)

    top_cat = max(spent, key=spent.get) if spent else "—"
    over_budget_count = sum(1 for cat, lim in budgets.items() if spent.get(cat, 0) > lim)

    left = (
        f"[dim]Current Month[/dim]  [bold cyan]{month}[/bold cyan]\n\n"
        f"[bright_green]Income  [/bright_green] ₹{income:>12,.2f}\n"
        f"[bright_red]Expenses[/bright_red] ₹{expense:>12,.2f}\n"
        f"[bold]Net     [/bold] [{'green' if net>=0 else 'red'}]₹{net:>12,.2f}[/]\n\n"
        f"Transactions: [cyan]{tx_count}[/cyan]"
    )
    right = (
        f"Top spend: [yellow]{top_cat}[/yellow]\n\n"
        f"Budgets over limit: [{'red' if over_budget_count else 'green'}]{over_budget_count}[/]\n\n"
        f"[dim]Use menu below to manage[/dim]"
    )

    console.print(Panel(
        Columns([left, right], padding=(0, 6)),
        title="[bold]💸 Finance Dashboard[/bold]",
        border_style="bright_cyan",
    ))


# ═══════════════════════════════════════════════════════════
#  MAIN MENU LOOP
# ═══════════════════════════════════════════════════════════

MENU = [
    ("1", "➕  Add transaction",         add_transaction_flow),
    ("2", "📋  View transactions",        view_transactions),
    ("3", "🗑️   Delete transaction",      delete_transaction_flow),
    ("4", "🎯  Manage budgets",           manage_budgets),
    ("5", "📊  Spending chart",           spending_chart),
    ("6", "📈  Income vs Expenses trend", trend_chart),
    ("7", "📄  Monthly report",           monthly_report),
    ("q", "🚪  Quit",                     None),
]


def print_menu():
    """Render the main menu as a Rich panel."""
    items = "\n".join(
        f"  [bold cyan]{key}[/bold cyan]  {label}"
        for key, label, _ in MENU
    )
    console.print(Panel(items, title="Menu", border_style="bright_blue", padding=(0, 2)))


def main():
    conn = init_db()

    while True:
        clear()
        dashboard(conn)
        print_menu()

        choice = Prompt.ask("\n[bold]Choose[/bold]").strip().lower()

        # Find matching menu entry
        action = None
        for key, _, fn in MENU:
            if choice == key:
                action = fn
                break

        if action is None and choice == "q":
            console.print("\n[bold green]Bye! Keep saving 💪[/bold green]\n")
            break
        elif action:
            action(conn)
            Prompt.ask("\n[dim]Press Enter to continue…[/dim]", default="")
        else:
            console.print("[red]Unknown option.[/red]")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()