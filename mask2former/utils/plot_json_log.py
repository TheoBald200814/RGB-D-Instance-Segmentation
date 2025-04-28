import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from collections import defaultdict
from math import ceil # Import ceil for calculating number of figures


def _extract_metrics_from_file_v1(json_file_path):
    """Helper function to extract metrics from a single JSON file."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {json_file_path}. Skipping.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {json_file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred while reading {json_file_path}: {e}. Skipping.")
        return None

    log_history = data.get("log_history")
    if not log_history:
        print(f"Warning: 'log_history' key not found or is empty in {json_file_path}. Skipping.")
        return None

    metrics = defaultdict(list)
    steps = defaultdict(list)

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue # Skip entries without a step

        # Training metrics
        if "loss" in entry and "learning_rate" in entry:
            metrics["train_loss"].append(entry.get("loss"))
            steps["train_loss"].append(step)
            metrics["lr"].append(entry.get("learning_rate"))
            steps["lr"].append(step)
            metrics["grad_norm"].append(entry.get("grad_norm")) # Might be None
            steps["grad_norm"].append(step)


        # Evaluation metrics
        elif "eval_loss" in entry:
             metrics["eval_loss"].append(entry.get("eval_loss"))
             steps["eval_loss"].append(step)
             metrics["eval_map"].append(entry.get("eval_map"))
             steps["eval_map"].append(step)
             metrics["eval_map_50"].append(entry.get("eval_map_50"))
             steps["eval_map_50"].append(step)
             metrics["eval_map_75"].append(entry.get("eval_map_75"))
             steps["eval_map_75"].append(step)
             # Extract other eval metrics here if needed


    # Filter out None values and ensure alignment (important for grad_norm mostly)
    filtered_metrics = {}
    for key, values in metrics.items():
        step_list = steps[key]
        # Filter based on the metric value being non-None
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if valid_indices: # Only add if there's actual data
             filtered_metrics[f'{key}_steps'] = [step_list[i] for i in valid_indices]
             filtered_metrics[key] = [values[i] for i in valid_indices]

    return filtered_metrics


def plot_multiple_training_metrics_separated_v1(json_file_paths, model_names, output_dir=None):
    """
    Loads training log history from multiple trainer_state.json files
    and plots key training and evaluation metrics for comparison,
    separating Train/Eval Loss and different mAP scores into individual plots.

    Args:
        json_file_paths (list[str]): List of paths to the trainer_state.json files.
        model_names (list[str]): List of names corresponding to each json file path.
                                 Used for labeling plots.
        output_dir (str, optional): Directory to save the plot image.
                                     If None, the plot will be displayed instead.
                                     Defaults to None.
    """
    if not isinstance(json_file_paths, list) or not isinstance(model_names, list):
        print("Error: json_file_paths and model_names must be lists.")
        return
    if len(json_file_paths) != len(model_names):
        print("Error: The number of json file paths must match the number of model names.")
        return
    if not json_file_paths:
        print("Error: No JSON file paths provided.")
        return

    all_metrics_data = {}
    for path, name in zip(json_file_paths, model_names):
        metrics = _extract_metrics_from_file_v1(path)
        if metrics: # Only store if data was successfully extracted
            all_metrics_data[name] = metrics

    if not all_metrics_data:
        print("Error: No valid data could be extracted from any of the provided files.")
        return

    # --- Plotting ---
    num_plots = 6 # TrainLoss, EvalLoss, LR/GradNorm, mAP, mAP50, mAP75
    # Increase figure height to accommodate more plots
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=True)
    fig.suptitle('Comparative Training Metrics (Separated)', fontsize=16)

    # Define color cycle for better distinction if many models
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    num_colors = len(colors)
    markers = ['o', 's', '^', 'd', 'v', '*', 'p', 'X'] # More markers


    # --- Plot 1: Train Loss ---
    ax = axs[0]
    for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
        color = colors[i % num_colors]
        if 'train_loss' in metrics and metrics['train_loss']:
            ax.plot(metrics.get('train_loss_steps', []), metrics['train_loss'],
                         label=f'{model_name}', alpha=0.9, color=color) # Simpler label now
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, linestyle=':')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # --- Plot 2: Eval Loss ---
    ax = axs[1]
    for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
        color = colors[i % num_colors]
        marker = markers[i % len(markers)]
        if 'eval_loss' in metrics and metrics['eval_loss']:
            ax.plot(metrics.get('eval_loss_steps', []), metrics['eval_loss'],
                         label=f'{model_name}', marker=marker, linestyle='--',
                         alpha=0.8, color=color) # Simpler label
    ax.set_ylabel('Loss')
    ax.set_title('Evaluation Loss')
    ax.grid(True, linestyle=':')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


    # --- Plot 3: Learning Rate and Gradient Norm ---
    ax_lr = axs[2]
    ax_gn = ax_lr.twinx() # Create a twin y-axis for grad norm
    lines_lr, labels_lr = [], []
    lines_gn, labels_gn = [], []

    for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
        color = colors[i % num_colors]
        # Plot Learning Rate
        if 'lr' in metrics and metrics['lr']:
            line, = ax_lr.plot(metrics.get('lr_steps', []), metrics['lr'],
                              label=f'{model_name} - LR', color=color, alpha=0.9)
            lines_lr.append(line)
            labels_lr.append(f'{model_name} - LR')

        # Plot Grad Norm
        if 'grad_norm' in metrics and metrics['grad_norm']:
             line, = ax_gn.plot(metrics.get('grad_norm_steps', []), metrics['grad_norm'],
                              label=f'{model_name} - Grad Norm', color=color, linestyle=':', alpha=0.6)
             lines_gn.append(line)
             labels_gn.append(f'{model_name} - Grad Norm')

    ax_lr.set_ylabel('Learning Rate', color='tab:blue') # Assume LR axis is primary blue
    ax_lr.tick_params(axis='y', labelcolor='tab:blue')
    ax_lr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax_gn.set_ylabel('Gradient Norm', color='tab:red') # Assume GN axis is secondary red
    ax_gn.tick_params(axis='y', labelcolor='tab:red')

    ax_lr.set_title('Learning Rate and Gradient Norm')
    # Combine legends from both axes and place outside
    ax_gn.legend(lines_lr + lines_gn, labels_lr + labels_gn, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_lr.grid(True, linestyle=':')


    # --- Plot 4: Evaluation mAP ---
    ax = axs[3]
    for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
        color = colors[i % num_colors]
        marker = markers[i % len(markers)]
        if 'eval_map' in metrics and metrics['eval_map']:
            ax.plot(metrics.get('eval_map_steps', []), metrics['eval_map'],
                         label=f'{model_name}', marker=marker, linestyle='-',
                         alpha=0.8, color=color) # Simpler label
    ax.set_ylabel('mAP Score')
    ax.set_title('Evaluation mAP')
    ax.grid(True, linestyle=':')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


    # --- Plot 5: Evaluation mAP@50 ---
    ax = axs[4]
    for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
        color = colors[i % num_colors]
        marker = markers[i % len(markers)]
        if 'eval_map_50' in metrics and metrics['eval_map_50']:
            ax.plot(metrics.get('eval_map_50_steps', []), metrics['eval_map_50'],
                         label=f'{model_name}', marker=marker, linestyle='-.', # Changed linestyle
                         alpha=0.8, color=color) # Simpler label
    ax.set_ylabel('mAP@50 Score')
    ax.set_title('Evaluation mAP@50')
    ax.grid(True, linestyle=':')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # --- Plot 6: Evaluation mAP@75 ---
    ax = axs[5]
    for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
        color = colors[i % num_colors]
        marker = markers[i % len(markers)]
        if 'eval_map_75' in metrics and metrics['eval_map_75']:
            ax.plot(metrics.get('eval_map_75_steps', []), metrics['eval_map_75'],
                         label=f'{model_name}', marker=marker, linestyle=':', # Changed linestyle
                         alpha=0.8, color=color) # Simpler label
    ax.set_ylabel('mAP@75 Score')
    ax.set_title('Evaluation mAP@75')
    ax.grid(True, linestyle=':')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


    # Common X-axis label for the last plot
    axs[-1].set_xlabel('Training Step')

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.97]) # Adjust right margin for legends

    # --- Output ---
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "comparative_training_metrics_separated.png")
            plt.savefig(output_path, bbox_inches='tight') # Use bbox_inches to include legend
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_dir}: {e}")
            print("Displaying plot instead.")
            plt.show()
    else:
        plt.show()

    plt.close(fig) # Close the figure to free memory


def _extract_metrics_from_file_v2(json_file_path):
    """
    Helper function to extract metrics from a single JSON file.
    Assumes the file is a dictionary with a 'log_history' key containing a list of dictionaries.
    Uses 'step' as the step value.
    Collects standard metrics and all 'eval_map_category' metrics.
    Filters out None values and ensures alignment.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found at {json_file_path}. Skipping.")
        return None
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {json_file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Warning: An unexpected error occurred while reading {json_file_path}: {e}. Skipping.")
        return None

    log_history = data.get("log_history")
    if not log_history or not isinstance(log_history, list):
        print(f"Warning: 'log_history' key not found or is not a list in {json_file_path}. Skipping.")
        return None

    metrics = defaultdict(list)
    steps = defaultdict(list)

    for entry in log_history:
        # Ensure entry is a dictionary
        if not isinstance(entry, dict):
            # print(f"Warning: Skipping non-dictionary log entry: {entry}")
            continue

        step = entry.get("step")
        if step is None:
            # Fallback to epoch if step is missing, or index if both are missing
            step = entry.get('epoch')
            if step is None:
                 # print(f"Warning: Skipping entry without 'step' or 'epoch': {entry}")
                 continue # Skip entries without a valid step

        # Ensure step is a valid number before proceeding
        if not isinstance(step, (int, float)):
             # print(f"Warning: Invalid step value '{step}'. Skipping entry for metric collection.")
             continue


        # Training metrics (collected if 'loss' and 'learning_rate' are present)
        if "loss" in entry and "learning_rate" in entry:
            # Append values directly, filtering will happen later
            metrics["train_loss"].append(entry.get("loss"))
            steps["train_loss"].append(step)
            metrics["lr"].append(entry.get("learning_rate"))
            steps["lr"].append(step)
            metrics["grad_norm"].append(entry.get("grad_norm")) # grad_norm might be None
            steps["grad_norm"].append(step)


        # Evaluation metrics (collected if 'eval_loss' is present)
        # This block will also collect category mAPs
        if "eval_loss" in entry:
             # Append values directly, filtering will happen later
             metrics["eval_loss"].append(entry.get("eval_loss"))
             steps["eval_loss"].append(step)

             # --- ADDED: Extract category specific mAPs ---
             for key, value in entry.items():
                 # Check if the key starts with 'eval_map_' and is NOT one of the overall ones
                 # Also exclude size-specific mAPs like _large, _medium, _small
                 if key.startswith('eval_map_') and key not in ['eval_map', 'eval_map_50', 'eval_map_75', 'eval_map_large', 'eval_map_medium', 'eval_map_small']:
                     metrics[key].append(value) # Append value (might be None)
                     steps[key].append(step)   # Append step
             # --- END ADDED ---

             # We will ignore overall eval_map, eval_map_50, eval_map_75 as per requirement
             # metrics["eval_map"].append(entry.get("eval_map"))
             # steps["eval_map"].append(step)
             # metrics["eval_map_50"].append(entry.get("eval_map_50"))
             # steps["eval_map_50"].append(step)
             # metrics["eval_map_75"].append(entry.get("eval_map_75"))
             # steps["eval_map_75"].append(step)


    # Filter out None values and ensure alignment for ALL collected metrics
    filtered_metrics = {}
    # Iterate through the keys that actually have collected data
    for key in metrics.keys():
        values = metrics[key]
        step_list = steps[key]

        # Filter based on the metric value being non-None
        # Ensure step_list has enough elements to match values (should be true with logic above, but defensive)
        valid_indices = [i for i, v in enumerate(values) if v is not None and i < len(step_list)]

        if valid_indices: # Only add if there's actual valid data points
             filtered_metrics[f'{key}_steps'] = [step_list[i] for i in valid_indices]
             filtered_metrics[key] = [values[i] for i in valid_indices]
        # If no valid data points, this metric and its steps are not added to filtered_metrics

    return filtered_metrics


def save_or_show_plot(fig, output_dir, base_filename, part_idx):
    """Helper function to save or show a matplotlib figure."""
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Construct filename like base_filename_partX.png
            name, ext = os.path.splitext(base_filename)
            output_filename = f"{name}_part{part_idx}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            fig.savefig(output_path, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot to {output_dir}: {e}")
            print("Displaying plot instead.")
            plt.show() # Use plt.show() which works for the current figure
    else:
        plt.show() # Display the current figure

    plt.close(fig) # Close the figure to free memory


def plot_multiple_training_metrics_with_category_map(json_file_paths, model_names, output_dir=None, max_category_plots_per_figure=25):
    """
    Loads training log history from multiple trainer_state.json files
    and plots key training and evaluation metrics for comparison,
    including separate plots for mAP of each category, split across multiple figures.

    Args:
        json_file_paths (list[str]): List of paths to the trainer_state.json files.
        model_names (list[str]): List of names corresponding to each json file path.
                                 Used for labeling plots.
        output_dir (str, optional): Directory to save the plot image.
                                     If None, the plot will be displayed instead.
                                     Defaults to None.
        max_category_plots_per_figure (int): Maximum number of category mAP plots
                                             to include in a single figure (after fixed plots).
                                             Defaults to 25.
    """
    if not isinstance(json_file_paths, list) or not isinstance(model_names, list):
        print("Error: json_file_paths and model_names must be lists.")
        return
    if len(json_file_paths) != len(model_names):
        print("Error: The number of json file paths must match the number of model names.")
        return
    if not json_file_paths:
        print("Error: No JSON file paths provided.")
        return

    all_metrics_data = {}
    all_category_map_keys = set() # Collect all unique category mAP keys across all files

    for path, name in zip(json_file_paths, model_names):
        metrics = _extract_metrics_from_file_v2(path)
        if metrics: # Only store if data was successfully extracted
            all_metrics_data[name] = metrics
            # Identify category mAP keys in this model's metrics
            for key in metrics.keys():
                # Check if the key starts with 'eval_map_', is not an overall one,
                # AND does NOT end with '_steps'
                if key.startswith('eval_map_') and key not in ['eval_map', 'eval_map_50', 'eval_map_75'] and not key.endswith('_steps'):
                     all_category_map_keys.add(key)

    if not all_metrics_data:
        print("Error: No valid data could be extracted from any of the provided files.")
        return

    # Sort category keys alphabetically for consistent plot order
    sorted_category_map_keys = sorted(list(all_category_map_keys))
    num_category_plots = len(sorted_category_map_keys)

    # Fixed plots
    num_fixed_plots = 3 # TrainLoss, EvalLoss, LR/GradNorm

    # Calculate number of figures needed for category plots
    # Add 1 for the first figure which contains fixed plots + first batch of categories
    num_category_figures = ceil(num_category_plots / max_category_plots_per_figure) if num_category_plots > 0 else 0
    total_figures_needed = 1 if num_fixed_plots > 0 or num_category_plots > 0 else 0 # At least one figure if there's anything to plot
    if num_category_figures > 1:
        total_figures_needed = 1 + (num_category_figures - 1) # First figure + subsequent category figures


    if total_figures_needed == 0:
        print("No plottable metrics found.")
        return

    base_output_filename = "comparative_training_metrics_with_category_map.png"

    # --- Plotting Figure 1: Fixed plots + first batch of category plots ---
    num_categories_in_first_fig = min(num_category_plots, max_category_plots_per_figure)
    total_plots_in_first_fig = num_fixed_plots + num_categories_in_first_fig

    if total_plots_in_first_fig > 0:
        fig1, axs1 = plt.subplots(total_plots_in_first_fig, 1, figsize=(12, max(4, total_plots_in_first_fig * 3)), sharex=True)
        if total_plots_in_first_fig == 1:
            axs1 = [axs1] # Ensure axs1 is iterable even with one subplot
        fig1.suptitle('Comparative Training Metrics (Part 1)', fontsize=16)

        # Define color cycle and markers for this figure
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        num_colors = len(colors)
        markers = ['o', 's', '^', 'd', 'v', '*', 'p', 'X', '<', '>', 'P', '*'] # More markers


        # --- Plot 1: Train Loss ---
        ax = axs1[0]
        for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
            color = colors[i % num_colors]
            if 'train_loss' in metrics:
                ax.plot(metrics['train_loss_steps'], metrics['train_loss'],
                             label=f'{model_name}', alpha=0.9, color=color)
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, linestyle=':')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


        # --- Plot 2: Eval Loss ---
        ax = axs1[1]
        for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
            color = colors[i % num_colors]
            marker = markers[i % len(markers)]
            if 'eval_loss' in metrics:
                ax.plot(metrics['eval_loss_steps'], metrics['eval_loss'],
                             label=f'{model_name}', marker=marker, linestyle='--',
                             alpha=0.8, color=color)
        ax.set_ylabel('Loss')
        ax.set_title('Evaluation Loss')
        ax.grid(True, linestyle=':')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


        # --- Plot 3: Learning Rate and Gradient Norm ---
        ax_lr = axs1[2]
        ax_gn = ax_lr.twinx()
        lines_lr, labels_lr = [], []
        lines_gn, labels_gn = [], []

        for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
            color = colors[i % num_colors]
            if 'lr' in metrics:
                line, = ax_lr.plot(metrics['lr_steps'], metrics['lr'],
                                  label=f'{model_name} - LR', color=color, alpha=0.9)
                lines_lr.append(line)
                labels_lr.append(f'{model_name} - LR')

            if 'grad_norm' in metrics:
                 line, = ax_gn.plot(metrics['grad_norm_steps'], metrics['grad_norm'],
                                  label=f'{model_name} - Grad Norm', color=color, linestyle=':', alpha=0.6)
                 lines_gn.append(line)
                 labels_gn.append(f'{model_name} - Grad Norm')

        ax_lr.set_ylabel('Learning Rate', color='tab:blue')
        ax_lr.tick_params(axis='y', labelcolor='tab:blue')
        ax_lr.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        ax_gn.set_ylabel('Gradient Norm', color='tab:red')
        ax_gn.tick_params(axis='y', labelcolor='tab:red')

        ax_lr.set_title('Learning Rate and Gradient Norm')
        if lines_lr or lines_gn:
            ax_gn.legend(lines_lr + lines_gn, labels_lr + labels_gn, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_lr.grid(True, linestyle=':')


        # --- Plot first batch of Category mAPs ---
        for j in range(num_categories_in_first_fig):
            category_key = sorted_category_map_keys[j]
            ax = axs1[num_fixed_plots + j] # Get the correct subplot for this category
            category_name = category_key.replace('eval_map_', '')

            for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
                color = colors[i % num_colors]
                marker = markers[i % len(markers)]
                if category_key in metrics:
                    ax.plot(metrics[category_key + '_steps'], metrics[category_key],
                            label=f'{model_name}', marker=marker, linestyle='-',
                            alpha=0.8, color=color)

            ax.set_ylabel('mAP Score')
            ax.set_title(f'Evaluation mAP ({category_name})')
            ax.grid(True, linestyle=':')
            if ax.get_legend_handles_labels()[0]:
                 ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

        # Set Common X-axis label for the last plot in this figure
        axs1[-1].set_xlabel('Training Step')

        # Adjust layout and save/show
        plt.tight_layout(rect=[0, 0.03, 0.9, 0.97])
        save_or_show_plot(fig1, output_dir, base_output_filename, 1)


    # --- Plotting Subsequent Figures for remaining category plots ---
    # Start index for categories in the sorted list
    category_start_idx = num_categories_in_first_fig

    # Loop through remaining figures
    for fig_idx in range(1, num_category_figures): # fig_idx starts from 1 for Part 2, Part 3, etc.
        start_idx_in_sorted_list = num_categories_in_first_fig + (fig_idx - 1) * max_category_plots_per_figure
        end_idx_in_sorted_list = min(num_category_plots, start_idx_in_sorted_list + max_category_plots_per_figure)
        current_batch_category_keys = sorted_category_map_keys[start_idx_in_sorted_list:end_idx_in_sorted_list]
        num_plots_in_this_fig = len(current_batch_category_keys)

        if num_plots_in_this_fig > 0:
             fig, axs = plt.subplots(num_plots_in_this_fig, 1, figsize=(12, max(4, num_plots_in_this_fig * 3)), sharex=True)
             if num_plots_in_this_fig == 1:
                 axs = [axs] # Ensure axs is iterable
             fig.suptitle(f'Comparative Training Metrics (Part {fig_idx + 1})', fontsize=16)

             # Define color cycle and markers for this figure (can reuse or redefine)
             prop_cycle = plt.rcParams['axes.prop_cycle']
             colors = prop_cycle.by_key()['color']
             num_colors = len(colors)
             markers = ['o', 's', '^', 'd', 'v', '*', 'p', 'X', '<', '>', 'P', '*']


             # Plot category mAPs on axs[0] onwards
             for j, category_key in enumerate(current_batch_category_keys):
                 ax = axs[j] # Get the correct subplot for this category within this figure
                 category_name = category_key.replace('eval_map_', '')

                 for i, (model_name, metrics) in enumerate(all_metrics_data.items()):
                     color = colors[i % num_colors]
                     marker = markers[i % len(markers)]
                     if category_key in metrics:
                         ax.plot(metrics[category_key + '_steps'], metrics[category_key],
                                 label=f'{model_name}', marker=marker, linestyle='-',
                                 alpha=0.8, color=color)

                 ax.set_ylabel('mAP Score')
                 ax.set_title(f'Evaluation mAP ({category_name})')
                 ax.grid(True, linestyle=':')
                 if ax.get_legend_handles_labels()[0]:
                      ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

             # Set Common X-axis label for the last plot in this figure
             axs[-1].set_xlabel('Training Step')

             # Adjust layout and save/show
             plt.tight_layout(rect=[0, 0.03, 0.9, 0.97])
             save_or_show_plot(fig, output_dir, base_output_filename, fig_idx + 1)

# --- Example Usage (in __main__ block) ---

if __name__ == "__main__":
    # Check if running with command-line arguments
    if len(sys.argv) > 1:
        print("Running plotting script with command-line arguments.")
        parser = argparse.ArgumentParser(description="Plot training metrics from multiple JSON log files, including category mAP.")
        parser.add_argument("--json_files", required=True, nargs='+',
                            help="List of paths to the trainer_state.json log files.")
        parser.add_argument("--model_names", required=True, nargs='+',
                            help="List of names corresponding to each JSON file (for plot labels). Must match the number of --json_files.")
        parser.add_argument("--output_dir", default=None,
                            help="Directory to save the plot image. If not specified, the plot will be displayed.")

        args = parser.parse_args()

        if len(args.json_files) != len(args.model_names):
             print("Error: Number of --json_files must match number of --model_names.")
             sys.exit(1)

        json_paths = args.json_files
        names = args.model_names
        output = args.output_dir

    else:
        print("Running plotting script in IDE without command-line arguments. Using hardcoded defaults.")
        # If no arguments, manually set default values for IDE testing
        # --- 在这里设置你的默认参数值 ---
        # !!! IMPORTANT: 请根据你的实际情况修改下面的路径和名称 !!!
        # 假设你的输入文件就是你提供的 NYU-v3-ultra.json
        nyu_v3_s = "/Users/theobald/Downloads/NYU-v3-single.json"
        nyu_v3_m = "/Users/theobald/Downloads/NYU-v3-multi.json"
        nyu_v3_u = "/Users/theobald/Downloads/NYU-v3-ultra.json"

        json_paths = [nyu_v3_s, nyu_v3_m, nyu_v3_u]
        names = ['single', 'multi', 'ultra']

        output = '/Users/theobald/Documents/code_lib/python_lib/shrimpDetection/mask2former/experiments/finetuning' # 替换为你希望保存图表的目录，或者设置为 None 只显示

        # --- 默认参数设置结束 ---

    # Run the plotting function
    try:
        plot_multiple_training_metrics_with_category_map(
            json_file_paths=json_paths,
            model_names=names,
            output_dir=output
        )
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
        import traceback
        traceback.print_exc()
