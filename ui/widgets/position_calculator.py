"""
Position sizing calculator widget (P2-7)
帳戶規模、風險比例、Kelly Criterion、即時計算部位大小
"""
import streamlit as st
import logging

logger = logging.getLogger(__name__)


def render_position_calculator():
    """Render the position sizing calculator as an expandable sidebar or inline widget."""

    with st.expander("🧮 部位大小計算器", expanded=False):
        st.caption("根據風險承受度計算最佳部位大小")

        # --- Inputs ---
        col1, col2 = st.columns(2)
        with col1:
            account_size = st.number_input(
                "帳戶總資金 (TWD)",
                value=1_000_000,
                step=100_000,
                min_value=10_000,
                format="%d",
                key="pos_calc_account",
            )
            entry_price = st.number_input(
                "進場價格",
                value=100.0,
                step=1.0,
                min_value=0.01,
                format="%.2f",
                key="pos_calc_entry",
            )
        with col2:
            risk_pct = st.slider(
                "每筆風險 (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="pos_calc_risk",
                help="建議 1-2%，最多不超過 5%",
            )
            stop_price = st.number_input(
                "停損價格",
                value=92.0,
                step=1.0,
                min_value=0.01,
                format="%.2f",
                key="pos_calc_stop",
            )

        if stop_price >= entry_price:
            st.warning("⚠️ 停損價應低於進場價")
            return

        # --- Calculations ---
        risk_per_share = entry_price - stop_price
        risk_pct_per_share = risk_per_share / entry_price * 100
        max_risk_amount = account_size * (risk_pct / 100)
        shares = int(max_risk_amount / risk_per_share)
        # Taiwan stocks trade in lots of 1000 (整股)
        lots = shares // 1000
        shares_rounded = lots * 1000

        position_value = shares_rounded * entry_price
        position_pct = position_value / account_size * 100
        max_loss = shares_rounded * risk_per_share

        # Kelly Criterion (simplified)
        col_win, col_rr = st.columns(2)
        with col_win:
            win_rate = st.slider(
                "歷史勝率 (%)",
                min_value=20,
                max_value=80,
                value=50,
                step=5,
                key="pos_calc_winrate",
            )
        with col_rr:
            reward_risk = st.slider(
                "風報比 (R:R)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="pos_calc_rr",
            )

        w = win_rate / 100
        kelly_full = w - (1 - w) / reward_risk if reward_risk > 0 else 0
        kelly_half = kelly_full / 2  # Half Kelly (more conservative)
        kelly_shares = int((account_size * max(kelly_half, 0)) / entry_price) if entry_price > 0 else 0
        kelly_lots = kelly_shares // 1000
        kelly_shares_rounded = kelly_lots * 1000

        # --- Results Display ---
        st.markdown("---")
        st.markdown("#### 📊 計算結果")

        # KPI strip
        kpi_html = f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0">
            <div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
                        border-radius:8px;padding:10px;text-align:center">
                <div style="color:#94a3b8;font-size:0.7rem">建議股數 (風控法)</div>
                <div style="color:#00d4ff;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace">
                    {shares_rounded:,} 股
                </div>
                <div style="color:#64748b;font-size:0.65rem">{lots} 張</div>
            </div>
            <div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.2);
                        border-radius:8px;padding:10px;text-align:center">
                <div style="color:#94a3b8;font-size:0.7rem">投入金額</div>
                <div style="color:#f0f0f0;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace">
                    ${position_value:,.0f}
                </div>
                <div style="color:#64748b;font-size:0.65rem">佔帳戶 {position_pct:.1f}%</div>
            </div>
            <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
                        border-radius:8px;padding:10px;text-align:center">
                <div style="color:#94a3b8;font-size:0.7rem">最大虧損</div>
                <div style="color:#ef4444;font-size:1.3rem;font-weight:700;font-family:JetBrains Mono,monospace">
                    -${max_loss:,.0f}
                </div>
                <div style="color:#64748b;font-size:0.65rem">-{risk_pct:.1f}% 帳戶</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

        # Kelly Criterion
        kelly_color = "#22c55e" if kelly_full > 0 else "#ef4444"
        kelly_html = f"""
        <div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.15);
                    border-radius:8px;padding:10px;margin:8px 0">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="color:#94a3b8;font-size:0.7rem">Kelly Criterion (半 Kelly)</div>
                    <div style="color:{kelly_color};font-size:1.1rem;font-weight:700;font-family:JetBrains Mono,monospace">
                        {kelly_half*100:.1f}% → {kelly_shares_rounded:,} 股 ({kelly_lots} 張)
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="color:#64748b;font-size:0.65rem">Full Kelly: {kelly_full*100:.1f}%</div>
                    <div style="color:#64748b;font-size:0.65rem">建議使用半 Kelly 降低風險</div>
                </div>
            </div>
        </div>
        """
        st.markdown(kelly_html, unsafe_allow_html=True)

        # Detail breakdown
        with st.expander("📋 詳細計算過程"):
            st.markdown(f"""
            | 項目 | 數值 |
            |------|------|
            | 每股風險 | {risk_per_share:.2f} ({risk_pct_per_share:.1f}%) |
            | 最大風險金額 | {max_risk_amount:,.0f} TWD |
            | 精確股數 | {shares:,} 股 |
            | 取整 (千股) | {shares_rounded:,} 股 ({lots} 張) |
            | Kelly f* | {kelly_full*100:.2f}% |
            | Half Kelly f*/2 | {kelly_half*100:.2f}% |
            """)

            st.info(
                "💡 **風控法**：固定每筆交易最大虧損為帳戶的 "
                f"{risk_pct:.1f}%（={max_risk_amount:,.0f} TWD）\n\n"
                "💡 **Kelly Criterion**：根據歷史勝率和風報比計算數學上最佳的部位比例。"
                "實務上建議使用「半 Kelly」以降低波動。"
            )
