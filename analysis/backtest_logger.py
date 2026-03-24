"""回測日誌 — 策略決策記錄器，用於績效歸因和策略調整。"""
import json
import os
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class BacktestLogger:
    """
    Records strategy decisions during backtesting for post-analysis.

    Usage in strategy:
        logger = BacktestLogger("isaac_v37")
        logger.log_entry(date, ticker, signal_type, score, price, reason)
        logger.log_exit(date, ticker, exit_type, price, pnl_pct, reason)
        logger.log_regime_change(date, old_regime, new_regime, exposure)
        logger.save()  # writes to data/backtest_logs/
        summary = logger.get_summary()
    """

    VALID_SIGNAL_TYPES = {"A", "B", "C", "D", "E"}
    VALID_EXIT_TYPES = {"trail_stop", "ma_break", "signal_d", "time_stop"}

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.entries: list[dict] = []
        self.exits: list[dict] = []
        self.regime_changes: list[dict] = []
        self.daily_snapshots: list[dict] = []
        self._created_at = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_entry(
        self,
        date: str,
        ticker: str,
        signal_type: str,
        score: float,
        entry_price: float,
        reason: str = "",
    ) -> None:
        """Record an entry signal."""
        if signal_type not in self.VALID_SIGNAL_TYPES:
            logger.warning(
                "Unknown signal_type '%s' for %s on %s", signal_type, ticker, date
            )
        self.entries.append(
            {
                "date": str(date),
                "ticker": ticker,
                "signal_type": signal_type,
                "score": float(score),
                "entry_price": float(entry_price),
                "reason": reason,
            }
        )

    def log_exit(
        self,
        date: str,
        ticker: str,
        exit_type: str,
        exit_price: float,
        pnl_pct: float,
        holding_days: int = 0,
        reason: str = "",
    ) -> None:
        """Record an exit signal."""
        if exit_type not in self.VALID_EXIT_TYPES:
            logger.warning(
                "Unknown exit_type '%s' for %s on %s", exit_type, ticker, date
            )
        self.exits.append(
            {
                "date": str(date),
                "ticker": ticker,
                "exit_type": exit_type,
                "exit_price": float(exit_price),
                "pnl_pct": float(pnl_pct),
                "holding_days": int(holding_days),
                "reason": reason,
            }
        )

    def log_regime_change(
        self,
        date: str,
        old_regime: str,
        new_regime: str,
        old_exposure: float,
        new_exposure: float,
    ) -> None:
        """Record a market regime change."""
        self.regime_changes.append(
            {
                "date": str(date),
                "old_regime": old_regime,
                "new_regime": new_regime,
                "old_exposure": float(old_exposure),
                "new_exposure": float(new_exposure),
            }
        )

    def log_daily_snapshot(
        self,
        date: str,
        n_positions: int,
        total_exposure: float,
        top_holdings: list[dict] | None = None,
    ) -> None:
        """Record daily portfolio snapshot."""
        self.daily_snapshots.append(
            {
                "date": str(date),
                "n_positions": int(n_positions),
                "total_exposure": float(total_exposure),
                "top_holdings": top_holdings or [],
            }
        )

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """
        Return a comprehensive summary of the backtest log.

        Returns dict with keys:
            strategy_name, total_entries, total_exits,
            signal_breakdown, exit_breakdown,
            best_trade, worst_trade,
            avg_holding_by_signal, regime_distribution
        """
        summary: dict = {
            "strategy_name": self.strategy_name,
            "total_entries": len(self.entries),
            "total_exits": len(self.exits),
            "signal_breakdown": self._signal_breakdown(),
            "exit_breakdown": self._exit_breakdown(),
            "best_trade": self._best_trade(),
            "worst_trade": self._worst_trade(),
            "avg_holding_by_signal": self._avg_holding_by_signal(),
            "regime_distribution": self._regime_distribution(),
        }
        return summary

    def _signal_breakdown(self) -> dict:
        """Count, win rate, and avg return per signal type."""
        by_signal: dict[str, list[dict]] = defaultdict(list)
        for entry in self.entries:
            by_signal[entry["signal_type"]].append(entry)

        # Build a ticker->exit map for matching entries to exits
        exit_map = self._build_exit_map()

        result = {}
        for sig, entries_list in sorted(by_signal.items()):
            pnls = []
            for e in entries_list:
                ex = exit_map.get(e["ticker"])
                if ex is not None:
                    pnls.append(ex["pnl_pct"])

            count = len(entries_list)
            wins = sum(1 for p in pnls if p > 0)
            win_rate = (wins / len(pnls) * 100) if pnls else 0.0
            avg_return = (sum(pnls) / len(pnls)) if pnls else 0.0

            result[sig] = {
                "count": count,
                "matched_exits": len(pnls),
                "win_rate": round(win_rate, 1),
                "avg_return_pct": round(avg_return, 2),
            }
        return result

    def _exit_breakdown(self) -> dict:
        """Count and avg pnl per exit type."""
        by_type: dict[str, list[float]] = defaultdict(list)
        for ex in self.exits:
            by_type[ex["exit_type"]].append(ex["pnl_pct"])

        result = {}
        for etype, pnls in sorted(by_type.items()):
            avg_pnl = sum(pnls) / len(pnls) if pnls else 0.0
            result[etype] = {
                "count": len(pnls),
                "avg_pnl_pct": round(avg_pnl, 2),
            }
        return result

    def _best_trade(self) -> dict | None:
        if not self.exits:
            return None
        best = max(self.exits, key=lambda x: x["pnl_pct"])
        return {
            "ticker": best["ticker"],
            "pnl_pct": best["pnl_pct"],
            "exit_date": best["date"],
            "holding_days": best["holding_days"],
        }

    def _worst_trade(self) -> dict | None:
        if not self.exits:
            return None
        worst = min(self.exits, key=lambda x: x["pnl_pct"])
        return {
            "ticker": worst["ticker"],
            "pnl_pct": worst["pnl_pct"],
            "exit_date": worst["date"],
            "holding_days": worst["holding_days"],
        }

    def _avg_holding_by_signal(self) -> dict:
        """Average holding period grouped by the entry signal type."""
        # Match exits back to entries by ticker (last entry per ticker)
        entry_signal: dict[str, str] = {}
        for e in self.entries:
            entry_signal[e["ticker"]] = e["signal_type"]

        by_signal: dict[str, list[int]] = defaultdict(list)
        for ex in self.exits:
            sig = entry_signal.get(ex["ticker"])
            if sig and ex["holding_days"] > 0:
                by_signal[sig].append(ex["holding_days"])

        return {
            sig: round(sum(days) / len(days), 1) if days else 0
            for sig, days in sorted(by_signal.items())
        }

    def _regime_distribution(self) -> dict:
        """Approximate % of time spent in each regime from snapshots & changes."""
        if not self.regime_changes:
            return {}

        counts: dict[str, int] = defaultdict(int)
        for rc in self.regime_changes:
            counts[rc["new_regime"]] += 1

        total = sum(counts.values())
        return {
            regime: round(c / total * 100, 1) for regime, c in sorted(counts.items())
        }

    def _build_exit_map(self) -> dict[str, dict]:
        """Map ticker -> last exit record (for matching with entries)."""
        exit_map: dict[str, dict] = {}
        for ex in self.exits:
            exit_map[ex["ticker"]] = ex
        return exit_map

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, base_dir: str = "data/backtest_logs") -> str:
        """
        Write the full log to a JSON file.

        Returns:
            The file path that was written.
        """
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_{timestamp}.json"
        filepath = os.path.join(base_dir, filename)

        payload = {
            "strategy_name": self.strategy_name,
            "created_at": self._created_at,
            "saved_at": datetime.now().isoformat(),
            "entries": self.entries,
            "exits": self.exits,
            "regime_changes": self.regime_changes,
            "daily_snapshots": self.daily_snapshots,
            "summary": self.get_summary(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Backtest log saved to %s", filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "BacktestLogger":
        """Reconstruct a BacktestLogger from a previously saved JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(data.get("strategy_name", "unknown"))
        instance.entries = data.get("entries", [])
        instance.exits = data.get("exits", [])
        instance.regime_changes = data.get("regime_changes", [])
        instance.daily_snapshots = data.get("daily_snapshots", [])
        instance._created_at = data.get("created_at", "")
        return instance


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------


def render_backtest_log_summary(summary: dict) -> str:
    """
    Return formatted markdown text suitable for st.markdown().

    Args:
        summary: dict returned by BacktestLogger.get_summary()
    """
    lines: list[str] = []
    name = summary.get("strategy_name", "?")
    lines.append(f"## 回測日誌摘要 - {name}")
    lines.append("")
    lines.append(
        f"**進場次數:** {summary.get('total_entries', 0)} &nbsp; "
        f"**出場次數:** {summary.get('total_exits', 0)}"
    )
    lines.append("")

    # --- Signal breakdown ---
    sig = summary.get("signal_breakdown", {})
    if sig:
        lines.append("### 進場信號分布")
        lines.append("")
        lines.append("| 信號 | 次數 | 配對出場 | 勝率 | 平均報酬 |")
        lines.append("|:----:|-----:|---------:|-----:|---------:|")
        for s, info in sorted(sig.items()):
            lines.append(
                f"| {s} | {info['count']} | {info['matched_exits']} "
                f"| {info['win_rate']:.1f}% | {info['avg_return_pct']:+.2f}% |"
            )
        lines.append("")

    # --- Exit breakdown ---
    ex = summary.get("exit_breakdown", {})
    if ex:
        lines.append("### 出場類型分布")
        lines.append("")
        lines.append("| 類型 | 次數 | 平均損益 |")
        lines.append("|:-----|-----:|---------:|")
        for etype, info in sorted(ex.items()):
            lines.append(
                f"| {etype} | {info['count']} | {info['avg_pnl_pct']:+.2f}% |"
            )
        lines.append("")

    # --- Best / Worst ---
    best = summary.get("best_trade")
    worst = summary.get("worst_trade")
    if best or worst:
        lines.append("### 最佳 / 最差交易")
        lines.append("")
        if best:
            lines.append(
                f"- **最佳:** {best['ticker']} "
                f"{best['pnl_pct']:+.2f}% "
                f"(持有 {best['holding_days']} 天, 出場 {best['exit_date']})"
            )
        if worst:
            lines.append(
                f"- **最差:** {worst['ticker']} "
                f"{worst['pnl_pct']:+.2f}% "
                f"(持有 {worst['holding_days']} 天, 出場 {worst['exit_date']})"
            )
        lines.append("")

    # --- Avg holding by signal ---
    holding = summary.get("avg_holding_by_signal", {})
    if holding:
        lines.append("### 平均持有天數 (依信號)")
        lines.append("")
        parts = [f"**{sig}:** {days} 天" for sig, days in sorted(holding.items())]
        lines.append(" &nbsp;|&nbsp; ".join(parts))
        lines.append("")

    # --- Regime distribution ---
    regime = summary.get("regime_distribution", {})
    if regime:
        lines.append("### 市場環境分布")
        lines.append("")
        parts = [f"**{r}:** {pct}%" for r, pct in sorted(regime.items())]
        lines.append(" &nbsp;|&nbsp; ".join(parts))
        lines.append("")

    return "\n".join(lines)
