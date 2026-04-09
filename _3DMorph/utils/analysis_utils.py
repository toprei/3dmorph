import os
from dataclasses import dataclass
from matplotlib.ticker import PercentFormatter
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

def load_results(frameworks_to_filter: list[str] = None) -> dict[str, pd.DataFrame]:
    framework_names = [
        "3DMorph",
        "Repaint",
        "TRELLIS",
        "Hunyuan3D",
        "TripoSG",
        "3DMorph-BB",
        "Repaint-BB",
        "TRELLIS multi-view",
    ]
    save_dir = Path(__file__).parents[1] / "statistics_analysis" / "evaluation_results"
    if frameworks_to_filter is None:
        frameworks_to_filter = []

    def combine_dfs(framework_name: str) -> pd.DataFrame:
        all_dfs = [pd.read_csv(str(save_dir / framework_name / f"dataset{i}-results.csv")) for i in range(11)]
        return pd.concat(all_dfs, ignore_index=True)

    return {name: combine_dfs(name) for name in framework_names if name not in frameworks_to_filter}


@dataclass
class PlotParameters:
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    figsize: tuple[int, int] = (10, 6)
    fontsize_small: int = 16
    fontsize_large: int = 20
    fontsize_title: int = 22
    density: bool = False
    mean: bool = True
    median: bool = True
    std: bool = True
    grid: bool = True
    show: bool = True
    save_dir: str | None = None
    title: str | None = None
    x_label: str | None = None
    y_label: str | None = None
    label_pad: int = 4
    bar_col: str = "steelblue"
    edge_col: str = "none"
    mean_col: str = "red"
    median_col: str = "green"
    tick_axis: str = "both"  # "x" | "y" | "both"
    precision: int = 3
    n_bins: int | np.ndarray = 50  # Absolute Number | Absolute Bin Width
    number_mode: str = "round"  # "round" | "percent" | "raw"

    def copy(self) -> "PlotParameters":
        """Create a deep copy of the PlotParameters instance."""
        return copy.deepcopy(self)

def get_bin_array(x_lim: tuple[float, float], n_bins=30) -> np.ndarray:
    x_min, x_max = x_lim
    bin_width = (x_max - x_min) / n_bins
    return np.arange(x_min, x_max + bin_width, bin_width)

def plot_distribution(data: pd.Series | list[pd.Series] | dict[str, pd.Series], params: PlotParameters = PlotParameters(), save_name: str = None) -> tuple[float, float]:
    """
    Plot histogram showing the distribution of scores for a specific metric.
    Args:
        data: One or multiple series to plot side-by-side. If its a dict, the keys are interpreted as titles.
    """
    
    if isinstance(data, list):
        values = data
        titles = [None] * len(data)
    # In dict case, set titles with keys and data with values
    elif isinstance(data, dict):
        titles = list(data.keys())
        values = list(data.values())
    # In single case, wrap data into list
    else:
        titles = [None]
        values = [data]
    
    if not values:
        return

    n_plots = len(values)
    a, b = params.figsize
    _, axes = plt.subplots(1, n_plots, figsize=(a * n_plots, b), sharey=True)
    
    # Handle single plot case
    if n_plots == 1:
        axes = [axes]
    
    for _, (ax, title, val_series) in enumerate(zip(axes, titles, values)):
        title = title if title is not None else params.title
        # Create histogram
        bins = params.n_bins
        ax.hist(val_series, bins=bins, edgecolor=params.edge_col, alpha=0.7, color=params.bar_col, density=params.density)
        
        # Calculate statistics
        mean_val = val_series.mean()
        median_val = val_series.median()
        std_val = val_series.std()

        if params.number_mode == "raw":
            mean_label = f'Mean: {mean_val}'
            median_label = f'Median: {median_val}'
            std_label = f'Std: {std_val}'
        elif params.number_mode == "round":
            mean_label = f'Mean: {mean_val:.{params.precision}f}'
            median_label = f'Median: {median_val:.{params.precision}f}'
            std_label = f'Std: {std_val:.{params.precision}f}'
        elif params.number_mode == "percent":
            mean_val_percent = mean_val * 100
            mean_label = f'Mean: {mean_val_percent:.{params.precision}f}%'
            median_val_percent = median_val * 100
            median_label = f'Median: {median_val_percent:.{params.precision}f}%'
            std_val_percent = std_val * 100
            std_label = f'Std: {std_val_percent:.{params.precision}f}%'
        else:
            raise ValueError(f"Unknown number mode {params.number_mode}")
        
        if params.mean:
            ax.axvline(mean_val, color=params.mean_col, linestyle='--', linewidth=2, label=mean_label)
        if params.median:
            ax.axvline(median_val, color=params.median_col, linestyle='--', linewidth=2, label=median_label)
        if params.std:
            ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, color='gray', label=std_label)
        
            
        # Set axis domain
        if params.x_lim is not None:
            a, b = params.x_lim
            if params.number_mode == "percent":
                ax.xaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xlim(a, b)
        if params.y_lim is not None:
            a, b = params.y_lim
            ax.set_ylim(a, b)
        
        # Set axis labels & title
        if params.x_label is not None:
            ax.set_xlabel(params.x_label, fontsize=params.fontsize_large, labelpad=params.label_pad)
        if params.y_label is not None:
            ax.set_ylabel(params.y_label, fontsize=params.fontsize_large, labelpad=params.label_pad)
        if title is not None:
            ax.set_title(title, fontsize=params.fontsize_title, fontweight='bold')
            
        ax.legend(fontsize=params.fontsize_small)
        if params.grid:
            ax.grid(axis='y', alpha=0.3)

        if params.tick_axis == "both":
            ax.tick_params(axis=params.tick_axis, which='major', labelsize=params.fontsize_small)
        elif params.tick_axis == "x":
            ax.tick_params(axis="x", which='major', labelsize=params.fontsize_small)
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        else:
            ax.tick_params(axis="y", which='major', labelsize=params.fontsize_small)
            ax.tick_params(axis='x', which='both', left=False, labelleft=False)
    
    max_y = max(ax.get_ylim()[1] for ax in axes)
    plt.tight_layout()

    if save_name is not None:
        save_path = save_name if params.save_dir is None else os.path.join(params.save_dir, save_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    if params.show:
        plt.show()
    plt.close()
    return 0.0, max_y


def plot_scatter(dataframe: pd.DataFrame, column_x: str, column_y: str, params: PlotParameters = PlotParameters(), save_name: str = None):
    plt.figure(figsize=params.figsize)
    sns.regplot(data=dataframe, x=column_x, y=column_y, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title(f'{column_x} vs {column_y}')
    plt.xlabel(column_x)
    plt.ylabel(column_y)

    if save_name is not None:
        save_path = save_name if params.save_dir is None else os.path.join(params.save_dir, save_name)
        plt.savefig(save_path)
    if params.show:
        plt.show()
    plt.close()


def plot_framework_comparison(dataframe: pd.DataFrame | dict, params: PlotParameters = PlotParameters(), save_name: str = None, skip=None, custom_order=None):
    if skip is None:
        skip = []
    
    # Convert to DataFrame if needed
    if isinstance(dataframe, dict):
        df = pd.DataFrame.from_dict(dataframe, orient='index').reset_index()
        df = df.rename(columns={'index': 'Framework'})
    else:
        df = dataframe.copy()
    
    # Filter out skipped frameworks
    df = df[~df['Framework'].isin(skip)]

    metrics = [col for col in df.columns if col != 'Framework']
    
    if custom_order is None:
        custom_order = list(df['Framework'].values)
    frameworks = [x for x in custom_order if x in df['Framework'].values]

    fig, ax = plt.subplots(figsize=params.figsize)
    x = np.arange(len(metrics))
    width = 0.9 / len(frameworks)
    
    # Plot bars for each framework
    for i, framework in enumerate(frameworks):
        values = df.loc[df['Framework'] == framework, metrics].values[0]
        offset = width * i - (width * len(frameworks) / 2) + width / 2
        positions = x + offset
        bars = ax.bar(positions, values, width, label=framework, alpha=0.8)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                   f'{value:.3f}'[1:],
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    title = "Framework Comparison" if params.title is None else params.title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Set y-axis limits
    if params.y_lim is not None:
        y_min, y_max = params.y_lim
    else:
        all_values = df[metrics].values.flatten()
        y_min = all_values.min()
        y_max = all_values.max()
    
    ax.set_ylim(bottom=y_min, top=y_max * 1.05)
    ax.legend(title='Framework', loc='upper right', bbox_to_anchor=(1.15, 1.0), 
              framealpha=0.9, edgecolor='gray')
    
    plt.tight_layout()
    if params.show:
        plt.show()

    return fig, ax