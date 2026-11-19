from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


_OKABE_ITO = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#332288"]


def _get_palette(codes: list[str]) -> dict[str, str]:
    pal: dict[str, str] = {}
    idx = 0
    for c in codes:
        if c in pal:
            continue
        if str(c).upper() == "PURE":
            pal[c] = "#555555"
        else:
            pal[c] = _OKABE_ITO[idx % len(_OKABE_ITO)]
            idx += 1
    return pal


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _yerr(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if df.empty or col not in df.columns or not df[col].notna().any():
        return None
    return df[col].fillna(0.0).to_numpy(dtype=float)


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    png_p = out_dir / f"{stem}.png"
    pdf_p = out_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")
    plt.close(fig)
    return png_p


def _discover_solvent_prefixes_from_cols(df: pd.DataFrame, prefix: str, suffix: str) -> list[str]:
    found: list[str] = []
    for c in df.columns:
        if c.startswith(prefix) and c.endswith(suffix):
            mid = c[len(prefix) : len(c) - len(suffix)]
            if mid and mid not in {"primary", "secondary", "sol"}:
                found.append(mid)
    return found


def generate_all_figures(results: dict[str, Any], out_dir: Path) -> list[Path]:
    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    enr = results.get("enrichment_summary", pd.DataFrame())
    enr_pairs = results.get("enrichment_solvent_pairs_summary", pd.DataFrame())
    kbi = results.get("kbi_summary", pd.DataFrame())
    hb = results.get("hbond_summary", pd.DataFrame())
    net = results.get("network_summary", pd.DataFrame())
    orient = results.get("orientation_summary", pd.DataFrame())
    hist2d = results.get("orientation_hist2d", pd.DataFrame())
    strain = results.get("xtb_strain", results.get("strain_summary", pd.DataFrame()))
    dimer_xtb = results.get("dimer_xtb", results.get("xtb_dimers", pd.DataFrame()))
    dft_dimers = results.get("dft_dimers", pd.DataFrame())
    cp_val = results.get("cp_validation", pd.DataFrame())
    trimer_xtb = results.get("trimer_xtb", results.get("xtb_trimers", pd.DataFrame()))
    dft_trimers = results.get("dft_trimers", pd.DataFrame())
    nci_df = results.get("nci_igmh", pd.DataFrame())
    led_df = results.get("led_eda", pd.DataFrame())
    cosmors = results.get("cosmors", pd.DataFrame())
    trans = results.get("transport_summary", pd.DataFrame())

    all_codes: list[str] = []
    for df in (enr, kbi, hb, net, orient, trans, dimer_xtb, led_df, nci_df):
        if isinstance(df, pd.DataFrame) and not df.empty and "system_code" in df.columns:
            for c in df["system_code"].dropna().astype(str):
                if c not in all_codes:
                    all_codes.append(c)
    palette = _get_palette(all_codes)

    # Fig 1: Solvation Shell & Local Enrichment (agnostic over N solvents)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    if not enr_pairs.empty:
        solvs = enr_pairs["solvent"].unique().tolist()
        codes = sorted(enr_pairs["system_code"].unique().tolist())
        x = np.arange(len(codes))
        w = 0.7 / max(len(solvs), 1)
        for idx_s, solv in enumerate(solvs):
            sub_p = enr_pairs.loc[enr_pairs["solvent"] == solv].set_index("system_code").reindex(codes)
            off = (idx_s - 0.5 * (len(solvs) - 1)) * w
            axes[0].bar(x + off, sub_p["n_shell_mean"], width=w, yerr=sub_p.get("n_shell_sd"), capsize=3, label=solv, color=_OKABE_ITO[idx_s % len(_OKABE_ITO)])
        axes[0].set_xticks(x, codes)
        axes[0].set_ylabel("First-shell molecules / solute")
        axes[0].set_title("A. First Solvation Shell Coordination")
        axes[0].legend()

        prim = solvs[0]
        sub_prim = enr_pairs.loc[enr_pairs["solvent"] == prim].set_index("system_code").reindex(codes)
        axes[1].bar(x, sub_prim["enrichment_molar_mean"], yerr=sub_prim.get("enrichment_molar_sd"), capsize=4, color=[palette.get(c, "#444") for c in codes])
        axes[1].axhline(1.0, ls="--", color="k", lw=1)
        axes[1].set_xticks(x, codes)
        axes[1].set_ylabel(f"Local Enrichment $E_{{{prim}}}$ (molar)")
        axes[1].set_title(f"B. {prim} Local Enrichment")
    elif not enr.empty:
        codes = enr["system_code"].tolist()
        x = np.arange(len(codes))
        solv_tags = _discover_solvent_prefixes_from_cols(enr, "n_", "_mean")
        w = 0.7 / max(len(solv_tags), 1)
        for idx_s, st in enumerate(solv_tags):
            off = (idx_s - 0.5 * (len(solv_tags) - 1)) * w
            axes[0].bar(x + off, enr[f"n_{st}_mean"], width=w, yerr=enr.get(f"n_{st}_sd"), capsize=3, label=st.upper(), color=_OKABE_ITO[idx_s % len(_OKABE_ITO)])
        axes[0].set_xticks(x, codes)
        axes[0].set_ylabel("First-shell molecules / solute")
        axes[0].set_title("A. First Solvation Shell Coordination")
        if solv_tags:
            axes[0].legend()
        e_col = next((c for c in enr.columns if c.startswith("e_") and c.endswith("_molar_mean")), None)
        if e_col is None and "e_primary_molar_mean" in enr.columns:
            e_col = "e_primary_molar_mean"
        if e_col:
            e_sd = e_col.replace("_mean", "_sd")
            axes[1].bar(x, enr[e_col], yerr=enr.get(e_sd), capsize=4, color=[palette.get(c, "#444") for c in codes])
            axes[1].axhline(1.0, ls="--", color="k", lw=1)
            axes[1].set_xticks(x, codes)
            axes[1].set_ylabel("Local Enrichment (molar)")
            axes[1].set_title("B. Primary Solvent Enrichment")

    if not kbi.empty:
        g_cols = [c for c in kbi.columns if c.startswith("G_S_") and c.endswith("_cm3_mol") and c not in {"G_S_primary_cm3_mol", "G_S_secondary_cm3_mol"}]
        if not g_cols and "G_S_primary_cm3_mol" in kbi.columns:
            g_cols = [c for c in ("G_S_primary_cm3_mol", "G_S_secondary_cm3_mol") if c in kbi.columns]
        sub_k = kbi.loc[kbi["system_code"].str.upper() != "PURE"]
        if sub_k.empty:
            sub_k = kbi
        if not sub_k.empty and g_cols:
            xc = np.arange(len(sub_k))
            w = 0.7 / max(len(g_cols), 1)
            for idx_g, gc in enumerate(g_cols):
                tag = gc.replace("G_S_", "").replace("_cm3_mol", "")
                off = (idx_g - 0.5 * (len(g_cols) - 1)) * w
                axes[2].bar(xc + off, sub_k[gc], width=w, yerr=_yerr(sub_k, gc.replace("_cm3_mol", "_sd")), capsize=3, label=f"$G_{{S,\\mathrm{{{tag}}}}}$", color=_OKABE_ITO[idx_g % len(_OKABE_ITO)])
            axes[2].axhline(0.0, ls=":", color="k", lw=0.8)
            axes[2].set_xticks(xc, sub_k["system_code"])
            axes[2].set_ylabel(r"KBI $G^\infty$ ($\mathrm{cm^3\ mol^{-1}}$)")
            axes[2].set_title("C. Kirkwood–Buff Integrals (GvdV + Krüger)")
            axes[2].legend()
    saved.append(_save_fig(fig, out_dir, "Fig_1_solvation_enrichment"))

    # Fig 2: H-bond Network & Orientational Distribution
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    if not hb.empty and "sol_hbonds_per_mol_mean" in hb.columns:
        xc = np.arange(len(hb))
        axes[0].bar(xc, hb["sol_hbonds_per_mol_mean"], yerr=_yerr(hb, "sol_hbonds_per_mol_sd"), capsize=4, color=[palette.get(c, "#444") for c in hb["system_code"]])
        axes[0].set_xticks(xc, hb["system_code"])
        axes[0].set_ylabel("Mean H-bonds / solute")
        axes[0].set_title("A. Solute–Solvent Hydrogen Bonding")
    if not net.empty:
        xc = np.arange(len(net))
        axes[1].bar(xc - 0.18, net["mean_degree"], width=0.35, yerr=_yerr(net, "mean_degree_sd"), capsize=3, label="Mean degree", color="#009E73")
        ax1_r = axes[1].twinx()
        ax1_r.plot(xc + 0.18, net["giant_frac"], "o--", color="#D55E00", label="Giant comp. frac")
        axes[1].set_xticks(xc, net["system_code"])
        axes[1].set_ylabel("Mean H-bond Degree")
        ax1_r.set_ylabel("Giant Component Fraction", color="#D55E00")
        axes[1].set_title("B. Collective H-Bond Network")
    if not hist2d.empty:
        target_sys = str(hist2d["system_code"].iloc[0])
        target_grp = str(hist2d["solvent_group"].iloc[0])
        sub_h = hist2d.loc[(hist2d["system_code"] == target_sys) & (hist2d["solvent_group"] == target_grp)]
        if not sub_h.empty:
            piv = sub_h.groupby(["cos_theta", "r_nm"], as_index=False)["density"].mean().pivot(index="cos_theta", columns="r_nm", values="density")
            im = axes[2].imshow(piv.values, aspect="auto", origin="lower", extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()], cmap="viridis")
            fig.colorbar(im, ax=axes[2], label="Density")
            axes[2].set_xlabel(r"$r_{\mathrm{solute-polar}}$ (nm)")
            axes[2].set_ylabel(r"$\cos\theta$")
            axes[2].set_title(f"C. {target_sys}–{target_grp} Orientational Map")
    elif not orient.empty:
        xc = np.arange(len(orient))
        sdf_col = next((c for c in orient.columns if c.startswith("sdf_") and c.endswith("_max")), None)
        if sdf_col:
            axes[2].bar(xc, orient[sdf_col], color="#0072B2")
        axes[2].set_xticks(xc, orient["system_code"])
        axes[2].set_title("C. Peak 3D SDF Density")
    saved.append(_save_fig(fig, out_dir, "Fig_2_hbonds_network_orientation"))

    # Fig 3: Conformational Strain & Dimer Interaction Energy
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    if not strain.empty:
        col_y = "e_strain_relaxed_kcal" if "e_strain_relaxed_kcal" in strain.columns else "e_strain_relaxed_mean_kcal"
        if len(strain) > 5:
            sns.violinplot(data=strain, x="system_code", y=col_y, hue="system_code", palette=palette, legend=False, ax=axes[0])
        else:
            axes[0].bar(strain["system_code"], strain[col_y], color=[palette.get(c, "#444") for c in strain["system_code"]])
        axes[0].set_ylabel(r"$E_{\mathrm{strain}}$ (kcal mol$^{-1}$)")
        axes[0].set_title("A. Solute Conformational Strain (GFN2-xTB)")
    if not dimer_xtb.empty:
        if len(dimer_xtb) > 10:
            sns.boxplot(data=dimer_xtb, x="pair", y="de_int_kcal", ax=axes[1], color="#76B7B2")
            axes[1].tick_params(axis="x", rotation=30)
        else:
            axes[1].bar(dimer_xtb["pair"], dimer_xtb["de_int_kcal"], color="#0072B2")
            axes[1].tick_params(axis="x", rotation=30)
        axes[1].axhline(0.0, ls=":", color="k")
        axes[1].set_ylabel(r"$\Delta E_{\mathrm{int}}$ (kcal mol$^{-1}$)")
        axes[1].set_title("B. Pairwise Interaction Distribution (xTB)")
    if not dft_dimers.empty:
        axes[2].scatter(dft_dimers["de_int_kcal"], dft_dimers["de_int_r2scan3c_kcal"], c=[palette.get(c, "#444") for c in dft_dimers["system_code"]], s=45, label="r2SCAN-3c")
        if not cp_val.empty:
            axes[2].scatter(cp_val["de_xtb_kcal"], cp_val["de_wb97xd3_cp_kcal"], marker="^", c="black", s=55, label=r"$\omega$B97X-D3 CP")
        lims = [
            min(dft_dimers["de_int_kcal"].min(), dft_dimers["de_int_r2scan3c_kcal"].min()) - 1.0,
            max(dft_dimers["de_int_kcal"].max(), dft_dimers["de_int_r2scan3c_kcal"].max()) + 1.0,
        ]
        axes[2].plot(lims, lims, "k--", lw=1)
        axes[2].set_xlabel(r"GFN2-xTB $\Delta E_{\mathrm{int}}$ (kcal mol$^{-1}$)")
        axes[2].set_ylabel(r"DFT $\Delta E_{\mathrm{int}}$ (kcal mol$^{-1}$)")
        axes[2].set_title("C. xTB vs Dispersion-DFT Parity")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "DFT parity populated\nupon ORCA execution", ha="center", va="center")
        axes[2].set_title("C. xTB vs DFT Parity")
    saved.append(_save_fig(fig, out_dir, "Fig_3_strain_interaction_energies"))

    # Fig 4: Many-Body Cooperativity & NCI / IGMH
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    if not trimer_xtb.empty:
        if len(trimer_xtb) > 6:
            sns.violinplot(data=trimer_xtb, x="system_code", y="e_3body_kcal", hue="system_code", palette=palette, legend=False, ax=axes[0])
        else:
            axes[0].bar(trimer_xtb["system_code"], trimer_xtb["e_3body_kcal"], color=[palette.get(c, "#444") for c in trimer_xtb["system_code"]])
        if not dft_trimers.empty:
            for idx_s, (sc, grp) in enumerate(dft_trimers.groupby("system_code")):
                axes[0].scatter([sc] * len(grp), grp["e_3body_r2scan3c_kcal"], color="black", marker="D", s=35, zorder=5, label="r2SCAN-3c" if idx_s == 0 else None)
            axes[0].legend()
        axes[0].axhline(0.0, ls="--", color="k")
        axes[0].set_ylabel(r"$E_{\mathrm{3body}}$ (kcal mol$^{-1}$)")
        axes[0].set_title("A. Three-Body Cooperativity")

    if not nci_df.empty:
        xc = np.arange(len(nci_df))
        axes[1].bar(xc, nci_df["igmh_frac_hb"], label="H-bond (sl2r < -0.015)", color="#0072B2")
        axes[1].bar(xc, nci_df["igmh_frac_vdw"], bottom=nci_df["igmh_frac_hb"], label="vdW (-0.015..0.010)", color="#009E73")
        axes[1].set_xticks(xc, nci_df["cluster_id"], rotation=35, ha="right", fontsize=7.5)
        axes[1].set_ylabel("IGMH Grid Fraction")
        axes[1].set_title("B. Quantitative IGMH Fingerprint")
        axes[1].legend()

        plotted = 0
        for sc in nci_df["system_code"].dropna().unique().tolist():
            sub = nci_df.loc[nci_df["system_code"] == sc]
            sp = Path(sub.iloc[0]["igmh_scatter_path"])
            if sp.exists():
                arr = np.loadtxt(sp)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] == 4:
                    active = arr[:, 0] >= 1e-3
                    if np.any(active):
                        arr = arr[active]
                    y_plot = arr[:, 0]
                else:
                    y_plot = arr[:, 4]
                if len(arr) > 5000:
                    idx_s = np.linspace(0, len(arr) - 1, 5000, dtype=int)
                    arr = arr[idx_s]
                    y_plot = y_plot[idx_s]
                axes[2].scatter(arr[:, 3], y_plot, s=2, alpha=0.35, color=palette.get(sc, "#444"), label=sc)
                plotted += 1
        if plotted:
            axes[2].set_xlim(-0.06, 0.04)
            axes[2].set_ylim(0.0, 0.05)
            axes[2].set_xlabel(r"$\mathrm{sign}(\lambda_2)\rho$ (a.u.)")
            axes[2].set_ylabel(r"$\delta g^{\mathrm{inter}}$ (a.u.)")
            axes[2].set_title("C. IGMH Scatter Spectrum")
            axes[2].legend(markerscale=4)
    else:
        axes[1].set_title("B. Quantitative IGMH Fingerprint")
        axes[2].set_title("C. IGMH Scatter Spectrum")
    saved.append(_save_fig(fig, out_dir, "Fig_4_manybody_nci_igmh"))

    # Fig 5: Energy Decomposition Analysis (LED) & Optional Benchmark / Selectivity Integration
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    if not led_df.empty:
        xc = np.arange(len(led_df))
        w = 0.2
        axes[0].bar(xc - 1.5 * w, led_df["e_elstat_kcal"], width=w, label="Electrostatics", color="#0072B2")
        axes[0].bar(xc - 0.5 * w, led_df["e_exch_kcal"], width=w, label="Exchange", color="#56B4E9")
        axes[0].bar(xc + 0.5 * w, led_df["e_disp_kcal"], width=w, label="Dispersion", color="#009E73")
        axes[0].bar(xc + 1.5 * w, led_df["e_prep_kcal"], width=w, label="Elec. Prep.", color="#D55E00")
        axes[0].plot(xc, led_df["e_total_kcal"], "ko-", lw=1.5, label=r"Total $\Delta E_{\mathrm{int}}$")
        axes[0].axhline(0.0, ls=":", color="k")
        axes[0].set_xticks(xc, led_df["label"], rotation=20)
        axes[0].set_ylabel("Energy (kcal mol$^{-1}$)")
        axes[0].set_title("A. DLPNO-CCSD(T) Local Energy Decomposition")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "LED populated upon\nORCA DLPNO-CCSD(T) run", ha="center", va="center")
        axes[0].set_title("A. DLPNO-CCSD(T) Local Energy Decomposition")

    if not cosmors.empty and "solute_code" in cosmors.columns:
        merged = cosmors.copy()
        if not dimer_xtb.empty:
            mean_de = dimer_xtb.groupby("system_code", as_index=False)["de_int_kcal"].mean()
            merged = merged.merge(mean_de, left_on="solute_code", right_on="system_code", how="left")
        else:
            merged["de_int_kcal"] = np.nan

        xc = np.arange(len(merged))
        if "ln_gamma_inf" in merged.columns:
            axes[1].bar(xc - 0.18, merged["ln_gamma_inf"], width=0.35, color="#CC79A7", label=r"Benchmark $\ln\gamma^\infty$")
        axr = axes[1].twinx()
        if merged["de_int_kcal"].notna().any():
            axr.plot(xc + 0.18, merged["de_int_kcal"], "s-", color="#0072B2", label=r"Mean $\Delta E_{\mathrm{int}}$ (kcal/mol)")
        axes[1].axhline(0.0, ls=":", color="k")
        axes[1].set_xticks(xc, merged["solute_code"])
        axes[1].set_ylabel(r"Reference $\ln\gamma^\infty$", color="#CC79A7")
        axr.set_ylabel(r"Mean $\Delta E_{\mathrm{int}}$ (kcal mol$^{-1}$)", color="#0072B2")
        axes[1].set_title("B. Continuum Reference vs Atomistic Forces")
    elif not dimer_xtb.empty:
        mean_de = dimer_xtb.groupby("system_code", as_index=False)["de_int_kcal"].mean()
        xc = np.arange(len(mean_de))
        axes[1].bar(xc, mean_de["de_int_kcal"], color=[palette.get(c, "#444") for c in mean_de["system_code"]])
        axes[1].set_xticks(xc, mean_de["system_code"])
        axes[1].set_ylabel(r"Mean $\Delta E_{\mathrm{int}}$ (kcal mol$^{-1}$)")
        axes[1].set_title("B. Mean Solute–Solvent Interaction")
    else:
        axes[1].set_title("B. Continuum Reference vs Atomistic Forces")
    saved.append(_save_fig(fig, out_dir, "Fig_5_led_mechanism_cosmors"))

    # Supplementary Fig S1: KBI Convergence & Equilibration
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    kbi_curves = results.get("kbi_curves", pd.DataFrame())
    if not kbi_curves.empty:
        non_pure_c = kbi_curves.loc[kbi_curves["system_code"].str.upper() != "PURE"]
        src_c = non_pure_c if not non_pure_c.empty else kbi_curves
        first_sys = str(src_c["system_code"].iloc[0])
        first_rep = str(src_c["replica"].iloc[0])
        sub_c = src_c.loc[(src_c["system_code"] == first_sys) & (src_c["replica"] == first_rep)]
        for pair in sorted(sub_c["pair"].unique().tolist())[:4]:
            sub = sub_c.loc[sub_c["pair"] == pair]
            axes[0].plot(sub["r_nm"], sub["G_raw_cm3_mol"], "--", alpha=0.6, label=f"{pair} raw")
            axes[0].plot(sub["r_nm"], sub["G_gvdv_cm3_mol"], "-", lw=1.8, label=f"{pair} GvdV")
        axes[0].axhline(0.0, ls=":", color="k")
        axes[0].set_xlabel(r"$R$ (nm)")
        axes[0].set_ylabel(r"$G(R)$ ($\mathrm{cm^3\ mol^{-1}}$)")
        axes[0].set_title(f"A. Raw vs GvdV Running KBI ({first_sys} {first_rep})")
        axes[0].legend(fontsize=7.5)
    equil_df = results.get("equilibration", pd.DataFrame())
    if not equil_df.empty:
        xc = np.arange(len(equil_df))
        axes[1].bar(xc, equil_df["density_mean_eq_kg_m3"], yerr=equil_df["density_sd_eq_kg_m3"], capsize=3, color="#0072B2")
        axes[1].set_xticks(xc, [f"{r['system_code']}_{r['replica']}" for _, r in equil_df.iterrows()], rotation=30, ha="right")
        axes[1].set_ylabel(r"Density ($\mathrm{kg\ m^{-3}}$)")
        axes[1].set_title("B. Equilibrated Density Across Replicates")
    saved.append(_save_fig(fig, out_dir, "Fig_S1_equilibration_kbi_convergence"))

    # Supplementary Fig S2: Green-Kubo Tail Diagnostic & MSD Self-Diffusion
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    gk_curves = results.get("gk_curves", pd.DataFrame())
    if not gk_curves.empty:
        for sc, grp in gk_curves.groupby("system_code"):
            m_curve = grp.groupby("tau_ps", as_index=False)["eta_running_tail_mpa_s"].mean()
            axes[0].plot(m_curve["tau_ps"], m_curve["eta_running_tail_mpa_s"], color=palette.get(sc, "#444"), label=sc)
        axes[0].set_xlabel(r"Lag $\tau$ (ps)")
        axes[0].set_ylabel(r"$\eta_{\mathrm{tail}}$ (mPa$\cdot$s)")
        axes[0].set_title("A. Green–Kubo Viscous Tail Diagnostic")
        axes[0].legend()
    if not trans.empty:
        d_cols = [c for c in trans.columns if c.startswith("d_") and c.endswith("_1e5_cm2_s")]
        if d_cols:
            xc = np.arange(len(trans))
            w = 0.7 / max(len(d_cols), 1)
            for idx_d, dc in enumerate(d_cols):
                lbl = "Solute" if dc == "d_sol_1e5_cm2_s" else dc[2:-10].upper()
                sd_col = f"d_{dc[2:-10]}_sd"
                off = (idx_d - 0.5 * (len(d_cols) - 1)) * w
                axes[1].bar(xc + off, trans[dc].fillna(0.0), width=w, yerr=_yerr(trans, sd_col), capsize=2, label=lbl, color=_OKABE_ITO[idx_d % len(_OKABE_ITO)])
            axes[1].set_xticks(xc, trans["system_code"])
            axes[1].set_ylabel(r"Self-Diffusion $D$ ($10^{-5}\ \mathrm{cm^2\ s^{-1}}$)")
            axes[1].set_title("B. Einstein Self-Diffusion Coefficients")
            axes[1].legend()
    saved.append(_save_fig(fig, out_dir, "Fig_S2_transport_gk_msd"))

    # Supplementary Fig S3: Site-Specific Atomistic RDFs, 1D Orientation & Block Density Windows
    site_rdfs = results.get("site_rdfs", pd.DataFrame())
    orient_1d = results.get("orientation_1d", pd.DataFrame())
    eq_win = results.get("equilibration_windows", pd.DataFrame())
    if not (site_rdfs.empty and orient_1d.empty and eq_win.empty):
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
        if not site_rdfs.empty:
            avg_site = site_rdfs.groupby(["system_code", "site_pair", "r_nm"], as_index=False)["g_r"].mean()
            for idx_p, (sp_name, grp) in enumerate(avg_site.groupby("site_pair")):
                sc = grp["system_code"].iloc[0]
                ls = "-" if idx_p % 2 == 0 else "--"
                axes[0].plot(grp["r_nm"], grp["g_r"], ls=ls, color=palette.get(sc, "#444"), label=f"{sc}:{sp_name}", lw=1.3)
            axes[0].set_xlim(0.2, 0.9)
            axes[0].set_xlabel(r"$r$ (nm)")
            axes[0].set_ylabel(r"$g(r)$")
            axes[0].set_title("A. Key-Site Atomistic RDFs")
            axes[0].legend(fontsize=6.5, ncol=1)
        if not orient_1d.empty:
            first_grp = str(orient_1d["solvent_group"].iloc[0])
            sub_o = orient_1d.loc[orient_1d["solvent_group"] == first_grp]
            if not sub_o.empty:
                stats_o = (
                    sub_o.groupby(["system_code", "cos_theta"], as_index=False)
                    .agg(mean=("density", "mean"), sd=("density", "std"))
                )
                for sc, grp in stats_o.groupby("system_code"):
                    sd_vals = grp["sd"].fillna(0.0)
                    axes[1].plot(grp["cos_theta"], grp["mean"], color=palette.get(sc, "#444"), label=sc, lw=1.8)
                    axes[1].fill_between(grp["cos_theta"], grp["mean"] - sd_vals, grp["mean"] + sd_vals, color=palette.get(sc, "#444"), alpha=0.18)
                axes[1].axhline(0.5, ls=":", color="k", lw=0.8)
                axes[1].set_xlabel(rf"$\cos\theta_{{\mathrm{{{first_grp}}}}}$")
                axes[1].set_ylabel(r"$P(\cos\theta)$")
                axes[1].set_title(f"B. First-Shell {first_grp} Orientation")
                axes[1].legend()
        if not eq_win.empty:
            for (sc, rep), grp in eq_win.groupby(["system_code", "replica"]):
                ls = "-" if rep == "R01" else "--"
                axes[2].plot(grp["t_mid_ns"], grp["density_mean_kg_m3"], ls=ls, color=palette.get(sc, "#444"), label=f"{sc}_{rep}", lw=1.3)
            axes[2].set_xlabel("Time (ns)")
            axes[2].set_ylabel(r"Density ($\mathrm{kg\ m^{-3}}$)")
            axes[2].set_title("C. Time-Window Density Convergence")
            axes[2].legend(fontsize=6.8, ncol=2)
        saved.append(_save_fig(fig, out_dir, "Fig_S3_sites_orientation_windows"))

    return saved
