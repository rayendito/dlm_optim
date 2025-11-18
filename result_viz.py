# %% Cell 1
import wandb
import pandas as pd
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

def plot_heatmap(title, to_viz):
    # Convert nested dict → DataFrame
    df = pd.DataFrame(to_viz)  # runs as rows, metrics as columns
    df = df.sort_index()
    plt.figure(figsize=(10, 3))
    plt.imshow(df, cmap="YlGn_r", aspect="auto", vmin=0, vmax=3.7)
    plt.colorbar(label="Loss (lower = better)")

    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.title(title)
    plt.ylabel("Metric")
    plt.xlabel("Run")
    plt.tight_layout()
    plt.show()

def visualize_losses(title, run_path_list, columns_to_show):
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
    plot_heatmap(title, to_viz)

# %% duar
visualize_losses("General Losses of P experiments", P_RUNS, ["train_loss", "val_loss"])

# %% duar
visualize_losses("General Losses of P experiments", P_RUNS, ["0.15", "0.25", "0.5", "0.75", "0.95"])


# %% Cell 3
df.columns
# %% Cell 3
duar = DEFAULT_BASELINE + P_RUNS

print(run.summary)
