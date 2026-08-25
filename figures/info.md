# Figures

model-comparison.png
model-comparison Kendall's τ (95% bootstrap CI) across all five targets, one panel per protein. Compares Generic and target-specific (TS) AEV-PLIG::GATv2, Generic ECIF::RF, each fingerprint family's own best-threshold config (ECIF::RF/XGB/SVR), TS ECIF::RF, and FEP+ (the gold-standard physics-based reference, set apart at the bottom). The headline "which method wins" figure.

ts-heatmap.png
ts-heatmap Four-panel heatmap (PCC, Kendall's τ, RMSE, Overlap@5) with proteins on the rows and every target-specific model on the columns — ECIF/PLEC × RF/SVR/XGB, plus Generic and TS AEV-PLIG::GATv2. Every cell is a real point estimate (no bootstrap resampling), giving a compact at-a-glance comparison across the whole model zoo.

ts-heatmap-with-overlap10.png
ts-heatmap-with-overlap10 Exploratory variant of ts-heatmap.png with a 5th panel added for Overlap@10 (fraction of the true top-10 strongest binders recovered in the model's predicted top-10). Kept as a separate figure pending a decision on whether to fold it into the main heatmap.

scatter-predicted-vs-experimental.png
scatter-predicted-vs-experimental Predicted vs. experimental ΔG scatter, 3 rows × 5 columns: rows are AEV-PLIG::GATv2 (generic), TS ECIF::RF, and family-specific ECIF::RF at its own best similarity threshold; columns are the five proteins. Each panel reports PCC/Kτ/RMSE/Overlap@5/Overlap@10 and highlights the 5 true strongest binders in red against the y=x reference line.

scatter-predicted-vs-experimental-10.png
scatter-predicted-vs-experimental-10 An earlier draft of the scatter figure above (predates the Overlap@5/@10 rename and the current annotation layout — still labelled "Generic AEV-PLIG" / "Recall" and highlights the top-10 strongest binders instead of top-5). Superseded by scatter-predicted-vs-experimental.png; kept for reference only.

Threshold sweep & seed variance (ECIF::RF)
family-specific-threshold-kt.png
family-specific-threshold-kt Kendall's τ across the full training-set-similarity threshold sweep (0.0–0.9) for ECIF::RF, one panel per protein, with Generic/TS AEV-PLIG::GATv2 shown as reference lines/bands and training-set size overlaid as bars on a log secondary axis. Point = real KT, whiskers = genuine bootstrap 95% CI throughout.

family-specific-threshold-kt-small.png
family-specific-threshold-kt-small Same data and convention as family-specific-threshold-kt.png, laid out as a 3×2 grid instead of a 5-row stack — a more compact version for side-by-side or slide use.

family-specific-threshold-overlap.png
family-specific-threshold-overlap Same threshold-sweep design as the KT figure above, but for Overlap@5 instead of Kendall's τ.

family-specific-seed-spread.png
family-specific-seed-spread For MCL1 only (the one protein with a full threshold sweep across every fingerprint family): raw per-seed/fold Kendall's τ spread (min–max whiskers + mean) across thresholds, for all six ECIF/PLEC families plus AEV-PLIG. Shows seed-to-seed training variance directly, as opposed to bootstrap CIs.

Target-specific AEV-PLIG
ts-aevplig-boxplot.png
ts-aevplig-boxplot Box + strip plot of target-specific AEV-PLIG::GATv2's Kendall's τ across seed/fold replicates, one box per protein, with a dashed line marking Generic AEV-PLIG::GATv2's KT for reference — how much does fine-tuning on-target help, and how much does that help vary run to run.

Dataset & protein characteristics
figure2_combined.png
figure2_combined Three-panel dataset/target overview: (A) training database composition table (PDBbind/BindingNet/BindingDB → Augmented, plus the FEP test set) broken down per protein; (B) TM-align structural similarity histograms of each target against the training database; (C) binding-site and chemistry characteristic bars (best druggability, meaningful pockets, mean Tanimoto, cliff fraction, ΔG std) per protein.

protein_characteristics_bars.png
protein_characteristics_bars Small-multiples bar chart, one panel per characteristic (best druggability, mean druggability, mean Tanimoto, cliff fraction, ΔG std), each on its own natural y-scale with exact values labelled above every bar. The most reliable way to compare these five targets quantitatively.

protein_characteristics_radar.png
protein_characteristics_radar Five-panel radar plot, one polygon per protein, on four genuinely 0–1-scaled axes (best druggability, mean druggability, mean Tanimoto, cliff fraction) — no normalisation needed since all four are already comparable fractions/scores. Good for an at-a-glance shape comparison; see the bar chart above for exact values.

tm_align_histograms.png
tm_align_histograms TM-align structural similarity score distributions of each target protein against everything in the training database, one histogram per protein — how structurally novel (or not) each target is relative to what the models were trained on.