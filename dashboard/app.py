"""Saved-results dashboard: July projections and historical policy evaluation."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "outputs" / "tables"
ACTIONS = ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]
BUDGET, MAX_OFFER, MAX_PCT20, MIN_HOLD = 62_000, 0.60, 0.15, 0.05

st.set_page_config(page_title="PromoPilot July Policy", layout="wide")


@st.cache_data
def _read_csv(name, modified_ns):
    """The file timestamp invalidates cached values after a pipeline rebuild."""
    return pd.read_csv(T / name)


def table(name, optional=False):
    path = T / name
    if not path.exists():
        if optional:
            return pd.DataFrame()
        st.error(f"Required result is missing: {name}. Rebuild the pipeline outputs.")
        st.stop()
    return _read_csv(name, path.stat().st_mtime_ns)


def dollars(value, decimals=0):
    return "$" + f"{value:,.{decimals}f}"


pol = table("optimized_policy_detailed.csv")
act = table("policy_action_summary.csv")
cmp = table("baseline_policy_comparison.csv")
aud = table("customer_policy_audit.csv")
heldout = table("heldout_policy_comparison.csv", optional=True)
selection_path = ROOT / "outputs" / "models" / "policy_selection.json"
selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
selected_policy = selection.get("selected_policy", selection.get("selected_method", "see saved selection"))
forecast_model = selection.get("forecast_model", "see saved model comparison")
logistic_comparison = table("logistic_cp07_comparison.csv", optional=True)
logistic_intervals = table("logistic_cp07_paired_bootstrap.csv", optional=True)
profit_comparison = table("model_comparison_cp07.csv", optional=True)
milp = table("milp_policy_detailed.csv", optional=True)

spend = pol.estimated_promo_cost.sum()
incremental = pol.estimated_incremental_ncm.sum()
offered = pol.recommended_action.ne("NO_OFFER")
offer_share = offered.mean()
pct20_share = pol.recommended_action.eq("PCT_20").sum() / max(int(offered.sum()), 1)
holdout_share = pol.measurement_holdout.mean()

st.title("PromoPilot · July promotion policy")
st.caption("July amounts are model projections. Historical randomized-outcome evaluation is shown separately.")
st.write(f"Selected allocation: **{selected_policy}** · Profit forecast model: **{forecast_model}**")
st.info("The allocation rule is chosen on CP04–06 constrained policy value. The profit model is chosen "
        "on predictive error. The model-optimized allocation remains a separate challenger; maximizing "
        "its forecast does not independently establish profitable targeting.")

tab1, tab2, tab3, tab4, tab_models, tab5 = st.tabs(
    ["Overview", "Policy mix", "Projected policies", "Historical evaluation", "Model performance", "Customer lookup"]
)

with tab1:
    c = st.columns(5)
    c[0].metric("Projected incremental NCM", dollars(incremental))
    c[1].metric("Projected promotion spend", dollars(spend), f"limit {dollars(BUDGET)}", delta_color="off")
    c[2].metric("Offer share", f"{offer_share:.1%}", f"limit {MAX_OFFER:.0%}", delta_color="off")
    c[3].metric("PCT_20 share of offers", f"{pct20_share:.1%}", f"limit {MAX_PCT20:.0%}", delta_color="off")
    c[4].metric("Randomized controls", f"{holdout_share:.1%}", f"min {MIN_HOLD:.0%}", delta_color="off")
    st.progress(min(spend / BUDGET, 1.0), text=f"Projected budget used: {spend / BUDGET:.1%}")
    st.progress(min(offer_share / MAX_OFFER, 1.0), text=f"Offer cap used: {offer_share / MAX_OFFER:.1%}")
    st.caption("Budget feasibility uses predicted spend; it is separate from cost calibration.")
    if selected_policy == "RANDOM_FREE_SHIP":
        st.write("The selected rule randomly assigns free shipping within the offer cap, keeps the fixed "
                 "measurement controls, and respects the projected budget. Recipients are not ranked by predicted uplift.")
    if not milp.empty:
        aligned = pol.set_index("customer_id").recommended_action
        challenger = milp.set_index("customer_id").recommended_action.reindex(aligned.index)
        st.caption(f"The model-optimized challenger assigns a different action to {(aligned != challenger).sum():,} customers.")
    with st.expander("Interpretation and prospective measurement"):
        st.write(
            "July customers have no historical campaign exposure or available purchase in the prior 90 days. "
            "Imputation permits scoring but does not prove that historical effects transfer to this population."
        )
        st.write(
            "The default outcome counts refunds inside the 14-day window. Later-return sensitivity requires "
            "mature labels and a fitting cutoff. Historical profile snapshot availability remains an assumption."
        )
        st.write(
            "For the July test, compare all customers randomized to the policy arm—including its NO_OFFER "
            "customers—with randomized controls. Comparing only selected offer recipients with controls is biased."
        )
        st.write("The 5% control minimum is a constraint, not a statistical-power calculation.")

with tab2:
    a = act.set_index("action").reindex(ACTIONS)
    left, right = st.columns(2)
    left.subheader("Customers by action")
    left.bar_chart(a.customers)
    right.subheader("Projected incremental NCM by action ($)")
    right.bar_chart(a.expected_incremental_ncm)
    st.dataframe(
        a.style.format(
            {
                "share": "{:.1%}",
                "expected_incremental_ncm": dollars,
                "mean_incremental_ncm": lambda v: dollars(v, 2),
                "expected_promo_spend": dollars,
                "mean_purchase_probability": "{:.1%}",
            }
        ),
        width="stretch",
    )

with tab3:
    st.subheader("Projected incremental NCM versus all NO_OFFER ($)")
    st.caption(
        "Every policy is scored with the same fitted outcome models. These are forecasts under a shared "
        "model, not observed outcomes or independent estimates of policy superiority."
    )
    st.caption("SELECTED_POLICY is the rule used for July. MODEL_OPTIMIZED_MILP maximizes the model's projected value and is retained for comparison.")
    st.bar_chart(cmp.set_index("policy").total_expected_incremental_ncm)
    show = (
        ["policy"]
        + [f"n_{x}" for x in ACTIONS]
        + [
            "offer_share",
            "pct20_share_among_offered",
            "expected_promo_spend",
            "total_expected_ncm",
            "total_expected_incremental_ncm",
            "incremental_ncm_per_promo_dollar",
        ]
    )
    st.dataframe(
        cmp[[c for c in show if c in cmp]].style.format(
            {
                "offer_share": "{:.1%}",
                "pct20_share_among_offered": "{:.1%}",
                "expected_promo_spend": dollars,
                "total_expected_ncm": dollars,
                "total_expected_incremental_ncm": dollars,
                "incremental_ncm_per_promo_dollar": "{:.2f}",
            },
            na_rep="–",
        ),
        width="stretch",
        hide_index=True,
    )

with tab4:
    st.subheader("Historical value from randomized outcomes")
    st.caption(
        "Doubly robust (DR) estimates evaluate complete constrained policies; intervals describe incremental "
        "NCM per customer versus all NO_OFFER. An interval crossing zero is inconclusive evidence of gain. "
        "Displayed test intervals are nominal, not adjusted for choosing the strongest test result."
    )
    st.warning(
        "CP07 was inspected in the original analysis and audit. The revised selection uses earlier campaigns, "
        "but this remains retrospective evidence rather than a fresh confirmatory experiment."
    )
    if heldout.empty:
        st.info("Historical evaluation results are not available in this output snapshot. See docs/VALIDATION_REPORT.md.")
    else:
        view = heldout.copy()
        if "split" in view and view.split.nunique() > 1:
            split = st.selectbox("Evaluation campaign", sorted(view.split.dropna().astype(str).unique()))
            view = view[view.split.astype(str) == split]
        plot_cols = {"policy", "dr_incremental_ncm_per_customer", "ci_low", "ci_high"}
        if plot_cols.issubset(view.columns):
            plot = view.dropna(subset=list(plot_cols)).copy()
            if not plot.empty:
                point = plot.dr_incremental_ncm_per_customer.to_numpy(float)
                low = plot.ci_low.to_numpy(float)
                high = plot.ci_high.to_numpy(float)
                valid = np.isfinite(point) & np.isfinite(low) & np.isfinite(high)
                plot, point, low, high = plot.loc[valid], point[valid], low[valid], high[valid]
                if len(plot):
                    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(plot))))
                    yy = np.arange(len(plot))
                    ax.hlines(yy, low, high, color="#4a78a8", linewidth=2)
                    ax.scatter(point, yy, color="#164e83", zorder=3)
                    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
                    ax.set_yticks(yy, plot.policy)
                    ax.invert_yaxis()
                    ax.set_xlabel("Estimated incremental NCM per customer ($), with 95% interval")
                    ax.spines[["top", "right"]].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        preferred = [
            "split", "policy", "selected_for_july", "n", "dr_ncm_per_customer",
            "dr_incremental_ncm_per_customer", "ci_low", "ci_high",
            "hajek_incremental_ncm_per_customer", "estimated_promo_spend",
            "offered_share", "matched_rows", "effective_sample_size",
        ]
        columns = [c for c in preferred if c in view]
        columns += [c for c in view if c not in columns]
        st.dataframe(view[columns], width="stretch", hide_index=True)
        st.caption(
            "Selection and paired-improvement evidence are documented in docs/VALIDATION_REPORT.md. "
            "The development winner is selected by its point estimate; uncertainty is reported separately. "
            "The bootstrap re-estimates campaign action frequencies but conditions on fitted models and "
            "fixed allocations; it does not cover future campaign or population changes."
        )
    robustness_path = ROOT / "outputs" / "experiments" / "policy" / "stochastic_free_ship_robustness.csv"
    if robustness_path.exists():
        with st.expander("Free-shipping recipient randomization check"):
            st.write("This check averages over random recipient subsets while preserving the fixed controls. "
                     "It shows how much the seed-42 result depends on the particular recipients selected. "
                     "It was not used to change the selected policy.")
            st.dataframe(pd.read_csv(robustness_path), width="stretch", hide_index=True)

with tab_models:
    st.subheader("Logistic regression: purchase probabilities")
    st.caption("Both the original refit and the selected model train through CP06 and predict CP07. "
               "Average precision measures ranking among purchasers; log loss and Brier score measure probability accuracy. "
               "CP07 is a retrospective audit, and the models were selected on CP04–06.")
    if logistic_comparison.empty:
        st.info("Rebuild notebook 05 to display the model comparison.")
    else:
        fair = logistic_comparison[logistic_comparison.model.eq("original_fair_refit")]
        chosen = logistic_comparison[logistic_comparison.model.eq("selected")]
        if not fair.empty and not chosen.empty:
            cols = st.columns(3)
            for column, key, label, direction in zip(cols, ["pr_auc", "log_loss", "brier"],
                                                    ["Average precision", "Log loss", "Brier score"],
                                                    ["normal", "inverse", "inverse"]):
                column.metric(label, f"{chosen.iloc[0][key]:.5f}", f"{chosen.iloc[0][key]-fair.iloc[0][key]:+.5f} versus original refit", delta_color=direction)
        st.dataframe(logistic_comparison, width="stretch", hide_index=True)
        if not logistic_intervals.empty:
            st.caption("Paired differences below are selected minus original. Higher is better for AUC/average precision; lower is better for losses. "
                       "Intervals spanning zero do not establish a statistically clear improvement.")
            st.dataframe(logistic_intervals, width="stretch", hide_index=True)
    st.subheader("XGBoost: expected net contribution")
    st.caption("The selected two-part model combines purchase probability with expected signed profit among purchasers. "
               "The margin prediction is partially pooled toward the treatment's purchaser mean to reduce noise. "
               "Lower RMSE and MAE are better; higher R² is better. Predictive improvement and targeting improvement are evaluated separately.")
    if profit_comparison.empty:
        st.info("Rebuild notebook 06 to display the profit model comparison.")
    else:
        pooled = profit_comparison[profit_comparison.scope.eq("pooled")]
        original = pooled[pooled.model.eq("ORIGINAL_TLEARNER")]
        chosen = pooled[pooled.model.eq(forecast_model)]
        if not original.empty and not chosen.empty:
            cols = st.columns(3)
            for column, key, label, direction in zip(cols, ["rmse", "mae", "r2"], ["RMSE", "MAE", "R²"], ["inverse", "inverse", "normal"]):
                column.metric(label, f"{chosen.iloc[0][key]:.4f}", f"{chosen.iloc[0][key]-original.iloc[0][key]:+.4f} versus original", delta_color=direction)
        st.dataframe(profit_comparison, width="stretch", hide_index=True)
        st.caption("A low R² still signals limited ability to explain individual profit variation, even when prediction errors improve.")

with tab5:
    left, right = st.columns([1, 2])
    query = left.text_input("Customer ID contains")
    selected_actions = right.multiselect("Recommended action", ACTIONS, default=ACTIONS)
    cols = [
        "customer_id", "recommended_action", "predicted_purchase_probability", "predicted_ncm",
        "estimated_incremental_ncm", "estimated_promo_cost", "measurement_holdout", "customer_tier", "is_member",
    ]
    view = aud[aud.recommended_action.isin(selected_actions)]
    if query:
        view = view[view.customer_id.astype(str).str.contains(query.strip(), regex=False, na=False)]
    st.write(f"{len(view):,} customers")
    st.dataframe(
        view[[c for c in cols if c in view]].style.format(
            {
                "predicted_purchase_probability": "{:.1%}",
                "predicted_ncm": lambda v: dollars(v, 2),
                "estimated_incremental_ncm": lambda v: dollars(v, 2),
                "estimated_promo_cost": lambda v: dollars(v, 2),
            }
        ),
        width="stretch",
        hide_index=True,
    )
    if len(view) == 1:
        row = view.iloc[0]
        st.subheader(f"Customer {row.customer_id}: projected NCM under each action")
        st.bar_chart(pd.Series({x: row[f"predicted_ncm_{x}"] for x in ACTIONS}, name="projected NCM ($)"))
        st.caption("These are model estimates of alternative outcomes, not observed individual causal effects.")
