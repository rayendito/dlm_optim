# %% Cell 1
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
api = wandb.Api()

# %% Cell 2
DEFAULT_BASELINE = ["rayendito/dlm_optim/pah97uk3"]

P_RUNS = [
    "rayendito/dlm_optim/vv7agngo",
    "rayendito/dlm_optim/2gwj6fia",
    "rayendito/dlm_optim/7jrnv0en",
    "rayendito/dlm_optim/npsj511c",
    "rayendito/dlm_optim/ako761bu",
]

KAPPA_RUNS = [
    "rayendito/dlm_optim/3evv3mv2",
    "rayendito/dlm_optim/oprd2cu2",
    "rayendito/dlm_optim/axnonfy0",
    "rayendito/dlm_optim/jsv9vkpe",
]

# %% functions yagesya

RUN_METRICS = ["train_loss", "val_loss", "0.15", "0.25", "0.5", "0.75", "0.95"]
def get_lowest_from_history(run_history):
    # Ensure it's a DataFrame
    if not hasattr(run_history, "to_dict"):
        run_history = pd.DataFrame(run_history)
    return {
        col: run_history[col].min().item()
        for col in run_history.select_dtypes(include=[float, int]).columns
        if col in RUN_METRICS
    }

def plot_heatmap(title, to_viz, col_name_todelete, col_name_toreplace):
    # Convert nested dict → DataFrame
    df = pd.DataFrame(to_viz)  # runs as rows, metrics as columns
    df = df.sort_index()
    for cnd in col_name_todelete:
        df.columns = df.columns.str.replace(cnd, "", regex=False)
    for cnr in col_name_toreplace:
        df.columns = df.columns.str.replace(cnr[0], cnr[1], regex=False)
    plt.figure(figsize=(10, 3))
    cmap = plt.cm.YlGn_r
    norm = plt.Normalize(vmin=df.min().min(), vmax=df.max().max())

    im = plt.imshow(df, cmap=cmap, aspect="auto", norm=norm)
    plt.colorbar(label="Loss (lower = better)")
    # annotate each cell
    for i in range(df.shape[0]):          # rows (metrics)
        for j in range(df.shape[1]):      # columns (runs)
            val = df.iloc[i, j]
            color = cmap(norm(val))[:3]   # RGB color of cell
            # compute perceived brightness (luminance)
            brightness = np.dot(color, [0.299, 0.587, 0.114])
            text_color = "black" if brightness > 0.5 else "white"
            plt.text(j, i, f"{val:.2f}",
                     ha="center", va="center",
                     color=text_color, fontsize=9, fontweight="bold")

    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.title(title)
    plt.ylabel("Validation")
    plt.xlabel("Run")
    plt.tight_layout()
    plt.show()

def visualize_losses(title, run_path_list, columns_to_show, col_name_todelete, col_name_toreplace):
    to_viz = {}
    for run_path in run_path_list:
        run = api.run(run_path)
        history = run.history()  # if you want full data
        df = pd.DataFrame(history)
        best_runs = get_lowest_from_history(df)
        
        if columns_to_show is None:
            columns_to_show = list(best_runs.keys())
            
        filtered = {k: v for k, v in best_runs.items() if k in columns_to_show}
        to_viz[run.name] = filtered
    plot_heatmap(title, to_viz, col_name_todelete, col_name_toreplace)

# %% general P losses
visualize_losses(
    "General Losses of W experiments",
    DEFAULT_BASELINE + P_RUNS,
    ["train_loss", "val_loss"],
    col_name_todelete=["diffusion_", "_k20"],
    col_name_toreplace=[("p", "w")],
)

# %% evaluated on different scenarios
visualize_losses(
    "W experiments on Different Validation Scenarios",
    DEFAULT_BASELINE + P_RUNS,
    ["0.15", "0.25", "0.5", "0.75", "0.95"],
    col_name_todelete=["diffusion_", "_k20"],
    col_name_toreplace=[("p", "w")],
)


# %% evaluated on different scenarios
visualize_losses(
    r"$W=0.5$ experiment with different $\kappa$",
    DEFAULT_BASELINE + KAPPA_RUNS,
    ["0.15", "0.25", "0.5", "0.75", "0.95"],
    col_name_todelete=["diffusion_", "p0.5_","_k20"],
    col_name_toreplace=[("p", "w")],
)
# 
# %% Cell 3
df.columns
# %% Cell 3
duar = DEFAULT_BASELINE + P_RUNS

print(run.summary)
