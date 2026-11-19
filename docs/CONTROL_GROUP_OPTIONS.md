# Two choices for the July measurement holdout

The project requires at least 5% of the 2,776 July scoring customers to receive `NO_OFFER` for measurement, selected using a defensible approach. That is at least 139 customers. The wording does not specifically require uniform random selection.

These are **options for the project manager**, not implemented changes. The allocator revision retains the existing 139 uniform-random controls so its comparison with the previous allocator uses identical constraints.

| Option | Procedure | Advantages | Limitations |
|---|---|---|---|
| **Stratified random controls** | Group customers using pre-campaign predicted incremental NCM, then randomly reserve controls within each group. Unequal rates can reserve more lower-gain customers and fewer higher-gain customers, totaling 139. | Retains randomized comparisons within groups and can reduce expected profit sacrificed for measurement. | Some profitable offer opportunities are still withheld. Unequal rates require a stratified or appropriately weighted analysis; the existing unweighted estimator cannot simply be reused. Precision and projected allocation value must be evaluated before choosing the rates. |
| **Optimize first; reserve existing no-offer customers** | Allocate offers across the full population, then designate 139 customers already assigned `NO_OFFER` as the measurement cohort. | Does not sacrifice additional model-projected profit to create the holdout. | Those customers were selected for different predicted responses. A simple comparison with recipients does not establish campaign-wide causal lift. Suitable for descriptive monitoring; causal interpretation needs a justified alternative design or additional assumptions. |

## Illustrative stratified design

Using the current forecasts, a median split of best predicted incremental NCM is approximately **$0.940633 per customer over 14 days**. It creates 1,388 lower-gain and 1,388 higher-gain customers. This is a sampling-design choice, not an offer-eligibility threshold or proof of actual responsiveness.

One possible illustration is 100 randomly chosen controls from the lower half and 39 from the higher half. Control probabilities would be approximately 7.20% and 2.81%. These counts have **not** been established as optimal or adequately powered. The model and strata must be frozen before campaign outcomes; selection probabilities must be recorded, and the final allocation must satisfy all campaign constraints.

For measurement, compare the complete POLICY arm with CONTROL within each stratum, including policy-arm customers assigned no offer. Combine results according to the defined estimand and population weights. The actual campaign total must also account for the number assigned to the policy in each stratum.

## Financial comparison already calculated

Holding the current profit forecasts, promotion costs, and campaign limits fixed:

| Planning scenario | Projected incremental NCM |
|---|---:|
| Previous model-optimized allocation with the existing random controls | $2,501.78 |
| Best 1,665 free-shipping gains across all customers; reserve controls from remaining no-offer customers | $2,553.80 |
| Maximum recoverable projected gain from changing the existing holdout | **$52.01** |

The $52.01 is **not** a measured or simulated gain from stratified randomization. No July outcomes are available. All displayed financial values are model projections.

## Question for the project manager

> Should we use stratified random controls, potentially with fewer controls among customers predicted to benefit most, to support campaign-impact measurement? Or is selecting the measurement cohort entirely from customers already assigned no offer acceptable, provided we report the limitations of causal measurement?

References: [J-PAL randomization guidance](https://www.povertyactionlab.org/resource/randomization), [J-PAL analysis guidance](https://www.povertyactionlab.org/resource/data-analysis?lang=en). Project source: `PROJECT_PROMPT.md` in the supplied data package.
