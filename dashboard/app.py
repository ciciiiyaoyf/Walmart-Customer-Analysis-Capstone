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
purchase_headline = table("purchase_headline_metrics.csv", optional=True)
profit_comparison = table("model_comparison_cp07.csv", optional=True)
profit_bias = table("profit_bias_by_offer.csv", optional=True)
profit_calibration = table("profit_calibration_by_offer.csv", optional=True)
hurdle_purchase = table("hurdle_purchase_metrics.csv", optional=True)
purchaser_margin = table("purchaser_margin_metrics.csv", optional=True)
milp = table("milp_policy_detailed.csv", optional=True)
july_coverage = table("july_transaction_coverage.csv", optional=True)
july_reference = table("july_organic_reference_window.csv", optional=True)

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
            "July customers have no historical campaign exposure. Their available records contain zero completed "
            "purchases in the recent 30/90-day windows; their complete purchase histories for those windows are unknown. "
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
    with st.expander("Purchase history coverage"):
        st.write("The decision cutoff is June 20, 2025 at 10:00. The 90-day window starts March 22 at 10:00; "
                 "the 30-day window starts May 21 at 10:00. July scoring includes the cutoff timestamp.")
        st.write("The latest ordinary transaction supplied is dated January 9, 2025. That is an observed event, "
                 "not a verified coverage endpoint. Later campaign-attributed orders do not establish complete "
                 "ordinary purchase history. No complete-source coverage contract was supplied.")
        if july_coverage.empty:
            st.info("Rebuild Step 07 to display the per-customer coverage audit.")
        else:
            no_observed = july_coverage.observed_orders_90d.eq(0).sum()
            unknown = july_coverage.complete_orders_90d.isna().sum()
            st.write(f"Of {len(july_coverage):,} customers, {no_observed:,} have no observed completed orders in "
                     f"the recent 90 days, and {unknown:,} have unknown complete 90-day counts.")
            coverage_columns = {
                "customer_id": "Customer", "observed_orders_30d": "Observed orders, 30 days",
                "complete_orders_30d": "Complete orders, 30 days",
                "orders_30d_coverage_status": "30-day coverage", "observed_orders_90d": "Observed orders, 90 days",
                "complete_orders_90d": "Complete orders, 90 days", "orders_90d_coverage_status": "90-day coverage",
            }
            st.dataframe(july_coverage[list(coverage_columns)].rename(columns=coverage_columns),
                         width="stretch", hide_index=True)
            st.caption("Missing complete counts mean unknown; observed zero does not establish that no purchase occurred.")
        if not july_reference.empty:
            reference_buyers = july_reference.observed_organic_reference_orders.gt(0).sum()
            reference_orders = july_reference.observed_organic_reference_orders.sum()
            st.write(f"The older reference interval, October 12, 2024 through January 9, 2025, contains "
                     f"{reference_orders:,.0f} observed completed ordinary orders across {reference_buyers:,} "
                     "July customers. This is a stale audit interval, not a recent-window replacement or a model predictor.")
        st.caption("The coverage correction preserves all 43 model input values and their order. "
                   "It clarifies what the records establish; it adds no predictive accuracy gain.")

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
    st.info("The model comparisons below measure the earlier model revision. The subsequent purchase-history "
            "coverage correction preserves the model input values and adds no accuracy gain. The additional "
            "metrics diagnose those same predictions; adding a metric is not a model improvement.")
    st.subheader("Logistic regression baseline: purchase probabilities")
    st.caption("One logistic regression baseline trains through CP06 and predicts CP07 for evaluation. "
               "Average precision summarizes the precision-recall ranking. Recall is the fraction of actual "
               "purchasers classified positive at a frozen development-selected threshold. "
               "CP07 is a retrospective audit. The same model specification is refitted through CP07 for July scoring.")
    if purchase_headline.empty:
        st.info("Rebuild notebook 05 to display average precision and development-threshold recall.")
    else:
        baseline = purchase_headline.iloc[0]
        cols = st.columns(2)
        cols[0].metric("Average precision", f"{baseline.average_precision:.5f}")
        cols[1].metric("Recall", f"{baseline.recall:.2%}")
        headline_columns = ["model", "n", "positives", "average_precision", "recall", "precision",
                            "threshold", "predicted_positive_share", "tp", "fp", "fn", "tn"]
        st.dataframe(purchase_headline[[c for c in headline_columns if c in purchase_headline]],
                     width="stretch", hide_index=True)
        st.caption("The baseline's threshold maximizes F1 on its out-of-time CP04–06 predictions, before CP07 "
                   "evaluation. Recall must be read with precision and the classified-positive share: predicting "
                   "everyone as a purchaser would produce 100% recall. These classifications do not choose promotion recipients.")
    with st.expander("Supporting purchase-probability diagnostics"):
        st.write("Log loss, Brier score and ROC-AUC describe the baseline's probability and ranking quality. "
                 "They do not replace the headline average precision and recall metrics.")
        if not purchase_headline.empty:
            st.dataframe(purchase_headline[["model", "log_loss", "brier", "roc_auc"]],
                         width="stretch", hide_index=True)
    with st.expander("Archived logistic development experiment (optional)"):
        st.write("The original-versus-revised comparison records an earlier development experiment. "
                 "Only the revised specification is used as the project's logistic baseline. "
                 "The small average-precision change did not establish a clear improvement; recall at "
                 "the separately chosen thresholds declined. This experiment is not a second operational baseline.")
        if not logistic_comparison.empty:
            st.dataframe(logistic_comparison, width="stretch", hide_index=True)
        if not logistic_intervals.empty:
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

    st.subheader("Profit bias and calibration by offer")
    st.caption("Mean prediction bias is predicted minus observed NCM. Positive values mean overprediction; "
               "negative values mean underprediction. An overall average can hide opposite errors across offers.")
    if profit_bias.empty:
        st.info("Run the metric diagnostics to display profit bias and calibration.")
    else:
        chart = profit_bias[profit_bias.scope.isin(ACTIONS)].pivot(index="scope", columns="model", values="mean_prediction_bias")
        st.bar_chart(chart.reindex(ACTIONS), y_label="Mean prediction bias ($ per customer)")
        st.dataframe(profit_bias, width="stretch", hide_index=True)
        st.caption("Bias intervals are customer-bootstrap intervals conditional on fitted models. Offer groups and "
                   "prediction bands can be small; check sample counts before interpreting differences.")
    if not profit_calibration.empty:
        scope = st.selectbox("Profit calibration group", ["pooled"] + ACTIONS)
        bands = profit_calibration[profit_calibration.scope.eq(scope)]
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, part in bands.groupby("model", sort=False):
            part = part.sort_values("pred_mean")
            ax.plot(part.pred_mean, part.obs_mean, marker="o", label=name)
        limits = bands[["pred_mean", "obs_mean"]].to_numpy(float)
        if limits.size and np.isfinite(limits).any():
            low, high = np.nanmin(limits), np.nanmax(limits)
            if low == high:
                low, high = low-1, high+1
            ax.plot([low, high], [low, high], "--", color="#777777", label="predicted = observed")
        ax.set_xlabel("Mean predicted NCM ($)")
        ax.set_ylabel("Mean observed NCM ($)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(bands, width="stretch", hide_index=True)
        st.caption("Bands group similar predictions within each model and offer; tied predictions stay together. "
                   "These are retrospective diagnostics, not a calibration fit or a new model-selection step.")

    st.subheader("Two-part XGBoost: purchase and purchaser margin")
    st.caption("The purchase component is evaluated on randomized CP07 customers under their observed offers. "
               "The margin component is evaluated only among actual purchasers, using its blended conditional "
               "margin forecast before multiplication by purchase probability.")
    if hurdle_purchase.empty:
        st.info("Run the metric diagnostics to display the separate two-part model checks.")
    else:
        pooled_purchase = hurdle_purchase[hurdle_purchase.scope.eq("pooled")]
        if not pooled_purchase.empty:
            row = pooled_purchase.iloc[0]
            a, b = st.columns(2)
            a.metric("XGBoost purchase average precision", f"{row.average_precision:.5f}")
            b.metric("XGBoost purchase recall", f"{row.recall:.5f}")
        columns = ["scope", "n", "buyers", "average_precision", "recall", "precision", "threshold",
                   "predicted_positive_share", "tp", "fp", "fn", "tn"]
        st.dataframe(hurdle_purchase[[c for c in columns if c in hurdle_purchase]], width="stretch", hide_index=True)
        st.caption("One purchase threshold is frozen from earlier-only CP04–06 predictions of the selected "
                   "two-part configuration and applied to every offer. Its randomized evaluation population "
                   "differs from the logistic baseline population, so their headline scores are not a direct model ranking.")
    if not purchaser_margin.empty:
        st.dataframe(purchaser_margin, width="stretch", hide_index=True)
        st.caption("Here n counts actual purchasers. Negative margins are retained. These conditional-margin "
                   "errors cannot be compared directly with full-population expected-NCM errors. "
                   "Use the Historical evaluation tab for complete-policy incremental profit and uncertainty.")

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
