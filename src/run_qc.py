import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import mne
import plotly.express as px

def run_qc(site, subjects_dir):
    """
    Run SNR-based quality control for GPIAS and Dublin EEG paradigms.

    This function iterates over all subject folders belonging to the selected
    site, loads available epoched EEG files, computes evoked-response SNR values
    from predefined electrode and time-window selections, saves summary CSV files,
    and exports interactive HTML boxplots for visual QC.
    """

    ## setting
    subjects_dir = Path(subjects_dir)
    dublin_paradigms = ["omi", "xxxxx", "xxxxy"]
    gpias_event_ids = ["PO90_pre", "PO90_post"]

    omi_event_id = "Stimulus 4"
    xx_event_id = "Stimulus 1"
    xy_event_id = "Stimulus 11"

    site_map = {
        "austin":     "1",
        "dublin":     "2",
        "ghent":      "3",
        "illinois":   "4",
        "regensburg": "5",
        "tuebingen":  "6",
        "zuerich":    "7",
    }

    picks = [
        "FC1", "FCz", "FC2",
        "C1", "Cz", "C2",
        "CP1", "CPz", "CP2"
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    du_window = [90, 110]   # ms
    gp_window = [180, 200]  # ms

    if site == "zuerich":
        gp_window = [165, 185]  # ms

    output_dir = subjects_dir / "QC"
    output_dir.mkdir(exist_ok=True, parents=True)

    ## snr function
    def compute_snr(evoked, picks, window):
        """Compute SNR = RMS(signal window) / STD(pre-stimulus baseline)."""
        valid_picks = [ch for ch in picks if ch in evoked.ch_names]
        if len(valid_picks) == 0:
            return np.nan

        signal = evoked.get_data(
            picks=valid_picks,
            tmin=window[0] * 1e-3,
            tmax=window[1] * 1e-3
        )
        baseline = evoked.get_data(
            picks=valid_picks,
            tmin=None,
            tmax=0
        )

        signal_amp = np.sqrt(np.mean(signal ** 2))
        noise_amp = np.std(baseline)

        if noise_amp == 0:
            return np.nan

        return signal_amp / noise_amp


    ## main part
    site_code = site_map[site]
    gpias_rows = []
    dublin_rows = []

    for subject_dir in tqdm(sorted(subjects_dir.iterdir())):
        if not subject_dir.is_dir():
            continue
        if not subject_dir.name.startswith(site_code):
            continue

        subject_id = subject_dir.name
        epochs_dir = subject_dir / "epochs"

        ## gpias
        gpias_file = epochs_dir / "epochs-gpias.fif"
        if gpias_file.exists():
            try:
                epochs = mne.read_epochs(gpias_file, preload=False, verbose="ERROR")

                ev_pre = epochs[gpias_event_ids[0]].average()
                ev_post = epochs[gpias_event_ids[1]].average()

                snr_pre = compute_snr(ev_pre, picks, gp_window)
                snr_post = compute_snr(ev_post, picks, gp_window)
                snr_mean = np.nanmean([snr_pre, snr_post])

                gpias_rows.append({
                    "subject_id": subject_id,
                    "snr_pre": snr_pre,
                    "snr_post": snr_post,
                    "snr_mean": snr_mean
                })

            except Exception as e:
                print(f"[WARNING] GPIAS failed for {subject_id}: {e}")

        ## dublin
        snr_omi = np.nan
        snr_xx = np.nan
        snr_xy = np.nan

        ## omi
        omi_file = epochs_dir / "epochs-omi.fif"
        if omi_file.exists():
            try:
                epochs = mne.read_epochs(omi_file, preload=False, verbose="ERROR")
                ev = epochs[omi_event_id].average()
                snr_omi = compute_snr(ev, picks, du_window)
            except Exception as e:
                print(f"[WARNING] OMI failed for {subject_id}: {e}")

        # xx
        xx_file = epochs_dir / "epochs-xxxxx.fif"
        if xx_file.exists():
            try:
                epochs = mne.read_epochs(xx_file, preload=False, verbose="ERROR")
                try:
                    ev = epochs[xx_event_id].average()
                except Exception:
                    ev = epochs["Stimulus/S  1"].average()
                snr_xx = compute_snr(ev, picks, du_window)
            except Exception as e:
                print(f"[WARNING] XX failed for {subject_id}: {e}")

        # xy
        xy_file = epochs_dir / "epochs-xxxxy.fif"
        if xy_file.exists():
            try:
                epochs = mne.read_epochs(xy_file, preload=False, verbose="ERROR")
                try:
                    ev = epochs[xy_event_id].average()
                except Exception:
                    ev = epochs["Stimulus/S  11"].average()
                snr_xy = compute_snr(ev, picks, du_window)
            except Exception as e:
                print(f"[WARNING] XY failed for {subject_id}: {e}")

        if not np.all(np.isnan([snr_omi, snr_xx, snr_xy])):
            dublin_rows.append({
                "subject_id": subject_id,
                "snr_omi": snr_omi,
                "snr_xx": snr_xx,
                "snr_xy": snr_xy,
                "snr_mean": np.nanmean([snr_omi, snr_xx, snr_xy])
            })

    ## now concat
    gpias_df = pd.DataFrame(gpias_rows)
    dublin_df = pd.DataFrame(dublin_rows)

    if not gpias_df.empty:
        gpias_df = gpias_df.sort_values("subject_id").reset_index(drop=True)

    if not dublin_df.empty:
        dublin_df = dublin_df.sort_values("subject_id").reset_index(drop=True)

    gpias_df.to_csv(output_dir / f"{site}_gpias_snr.csv", index=False)
    dublin_df.to_csv(output_dir / f"{site}_dublin_snr.csv", index=False)

    ## html plots
    if not gpias_df.empty:
        gpias_long = gpias_df.melt(
            id_vars="subject_id",
            value_vars=["snr_pre", "snr_post", "snr_mean"],
            var_name="condition",
            value_name="snr"
        ).dropna()

        fig_gpias = px.box(
            gpias_long,
            x="condition",
            y="snr",
            color="condition",
            points="all",
            hover_data=["subject_id"],
            template="simple_white",
            title=f"GPIAS SNR ({site.title()})",
            color_discrete_sequence=colors
        )
        fig_gpias.update_traces(
                            jitter=0.2,
                            pointpos=0,
                            marker=dict(size=8, opacity=0.8),
                            line=dict(width=1.5)
                            )
        fig_gpias.update_traces(
            hovertemplate="<b>%{x}</b><br>SNR: %{y:.3f}<br>Subject: %{customdata[0]}<extra></extra>"
            )
        fig_gpias.update_layout(
            width=950,
            height=600,
            font=dict(size=15),
            showlegend=False,
            xaxis_title="",
            yaxis_title="SNR",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        gpias_html = output_dir / f"{site}_gpias_snr_boxplot.html"
        html_str = fig_gpias.to_html(include_plotlyjs="cdn", full_html=False)
        centered_html = f"""
                        <html>
                        <head></head>
                        <body style="display:flex; justify-content:center; align-items:center; height:100vh; margin:0;">
                            {html_str}
                        </body>
                        </html>
                        """
        with open(gpias_html, "w") as f:
            f.write(centered_html)

    if not dublin_df.empty:
        dublin_long = dublin_df.melt(
            id_vars="subject_id",
            value_vars=["snr_omi", "snr_xx", "snr_xy", "snr_mean"],
            var_name="condition",
            value_name="snr"
        ).dropna()

        fig_dublin = px.box(
            dublin_long,
            x="condition",
            y="snr",
            color="condition",
            points="all",
            hover_data=["subject_id"],
            template="simple_white",
            title=f"Dublin Paradigm SNR ({site.title()})",
            color_discrete_sequence=colors
        )
        fig_dublin.update_traces(
                            jitter=0.2,
                            pointpos=0,
                            marker=dict(size=8, opacity=0.8),
                            line=dict(width=1.5)
                            )
        fig_dublin.update_traces(
            hovertemplate="<b>%{x}</b><br>SNR: %{y:.3f}<br>Subject: %{customdata[0]}<extra></extra>"
            )
        fig_dublin.update_layout(
            width=950,
            height=600,
            font=dict(size=15),
            showlegend=False,
            xaxis_title="",
            yaxis_title="SNR",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        dublin_html = output_dir / f"{site}_dublin_snr_boxplot.html"
        html_str = fig_dublin.to_html(include_plotlyjs="cdn", full_html=False)
        centered_html = f"""
                        <html>
                        <head></head>
                        <body style="display:flex; justify-content:center; align-items:center; height:100vh; margin:0;">
                            {html_str}
                        </body>
                        </html>
                        """
        with open(dublin_html, "w") as f:
            f.write(centered_html)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute SNR QC metrics")
    parser.add_argument("site", type=str, help="Site name (e.g. zuerich)")
    parser.add_argument("subjects_dir", type=str, help="Path to subjects directory")
    args = parser.parse_args()
    run_qc(args.site, args.subjects_dir)