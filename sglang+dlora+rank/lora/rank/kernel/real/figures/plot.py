import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_fit_with_var_table(x, y_true, y_pred, xlabel, title, filename):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])

    # Main plot
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(x, y_true, label="Ground Truth", color="blue", alpha=0.7)
    ax0.plot(x, y_pred, label="Fitted Line", color="red", linewidth=2)
    ax0.set_xlabel(xlabel)
    ax0.set_ylabel("mean_ms")
    ax0.set_title(title)
    ax0.legend()

    # Build table: group by x, show pred, true max/min, (max-pred), (min-pred)
    df = pd.DataFrame({xlabel: x, "Ground Truth": y_true, "Predicted": y_pred})
    grouped = df.groupby(xlabel)
    table_data = []
    for val, group in grouped:
        pred_val = group["Predicted"].iloc[0]
        max_true = group["Ground Truth"].max()
        min_true = group["Ground Truth"].min()
        diff_max = max_true - pred_val
        diff_min = min_true - pred_val
        table_data.append([
            f"{val:.3f}",
            f"{pred_val:.3f}",
            f"{max_true:.3f}",
            f"{min_true:.3f}",
            f"{diff_max:.3f}",
            f"{diff_min:.3f}"
        ])
    col_labels = [xlabel, "Predicted", "True Max", "True Min", "Max-Pred", "Min-Pred"]

    # Only show first 10 and last 10 rows if too long
    if len(table_data) > 20:
        table_show = table_data[:10] + [["..."]*6] + table_data[-10:]
    else:
        table_show = table_data

    # Table on the right
    ax1 = fig.add_subplot(gs[1])
    ax1.axis('off')
    tbl = ax1.table(
        cellText=table_show,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 1.5)
    ax1.set_title("Per-variable True Value Range", pad=20)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Read data
triton = pd.read_csv("/workspace/sglang/benchmark/lora/rank/kernel/real/triton/triton_dec.csv")
csgmv = pd.read_csv("/workspace/sglang/benchmark/lora/rank/kernel/real/csgmv/csgmv_dec.csv")

# Triton fit
alpha_t = 0.0022599368185990654
beta_t = 17.987606140120654
x_t = triton["bs_x_max_rank"]
y_true_t = triton["mean_ms"]
y_pred_t = alpha_t * x_t + beta_t

plot_fit_with_var_table(
    x_t, y_true_t, y_pred_t,
    xlabel="bs_x_max_rank",
    title="Triton: Fit vs Ground Truth",
    filename="triton_latency_fit_var_table.png"
)

# CSGMV fit
alpha_c = 0.032860531060909304
beta_c = 17.55006135505361
x_c = csgmv["sqrt_bs_x_max_rank"]
y_true_c = csgmv["mean_ms"]
y_pred_c = alpha_c * x_c + beta_c

plot_fit_with_var_table(
    x_c, y_true_c, y_pred_c,
    xlabel="sqrt_bs_x_max_rank",
    title="CSGMV: Fit vs Ground Truth",
    filename="csgmv_latency_fit_var_table.png"
)