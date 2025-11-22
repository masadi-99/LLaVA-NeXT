import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
warnings.filterwarnings('ignore')

# =====================
# ECG plotting helpers (unchanged)
# =====================

def create_ecg_grid(ax, duration_s, y_low, y_high,
                    paper_speed_mm_s=25, gain_mm_mV=10):
    """Draw standard ECG paper background (minor 1 mm, major 5 mm)."""
    minor_time_s = 1.0 / paper_speed_mm_s
    major_time_s = 5.0 / paper_speed_mm_s
    minor_voltage_mV = 1.0 / gain_mm_mV
    major_voltage_mV = 5.0 / gain_mm_mV

    t = 0.0
    while t <= duration_s + 1e-12:
        ax.axvline(x=t, color='#ffb3b3', lw=0.3, alpha=0.7, zorder=0)
        t += minor_time_s
    v = y_low
    while v <= y_high + 1e-12:
        ax.axhline(y=v, color='#ffb3b3', lw=0.3, alpha=0.7, zorder=0)
        v += minor_voltage_mV

    t = 0.0
    while t <= duration_s + 1e-12:
        ax.axvline(x=t, color='#ff6666', lw=0.6, alpha=0.9, zorder=0)
        t += major_time_s
    v = y_low
    while v <= y_high + 1e-12:
        ax.axhline(y=v, color='#ff6666', lw=0.6, alpha=0.9, zorder=0)
        v += major_voltage_mV

    ax.set_facecolor('#fff5f5')

def draw_calibration_pulse(ax, start_s=0.2, height_mV=1.0, width_s=0.2):
    y0 = ax.get_ylim()[0]
    t0, t1 = start_s, start_s + width_s
    y_step = y0 + height_mV
    ax.plot([t0, t0, t1, t1], [y0, y_step, y_step, y0],
            color='black', lw=1.6, solid_joinstyle='miter', zorder=3)

def annotate_seconds(ax, duration_s, every_s=1.0):
    ticks = np.arange(0, duration_s + 1e-9, every_s)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks], fontsize=9)
    for t in ticks:
        ax.axvline(x=t, color='black', lw=0.6, alpha=0.25, zorder=1)

def add_six_second_bracket(ax, y_frac=0.1, start_s=0.0, width_s=6.0):
    y0, y1 = ax.get_ylim()
    y = y0 + (y1 - y0) * y_frac
    x0, x1 = start_s, start_s + width_s
    ax.plot([x0, x0, x1, x1], [y, y*0.999, y*0.999, y],
            color='black', lw=1.0, zorder=3)
    ax.text((x0+x1)/2, y, "6 s", ha='center', va='bottom', fontsize=9)

# =====================
# Main plotting API (unchanged)
# =====================

def plot_hospital_ecg(ecg_data,
                      sampling_rate=500,
                      paper_speed=25,
                      amplitude_scale=10,
                      lead_names=None,
                      patient_info=None,
                      save_path=None,
                      rhythm_lead_idx=1,
                      rhythm_duration_s=10):

    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    n_samples = ecg_data.shape[1]
    duration_s = n_samples / float(sampling_rate)
    t = np.linspace(0, duration_s, n_samples)

    mn, mx = np.nanmin(ecg_data), np.nanmax(ecg_data)
    pad = 0.1 * (mx - mn + 1e-12)
    y_low, y_high = mn - pad, mx + pad

    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.8],
                          hspace=0.15, wspace=0.1,
                          left=0.05, right=0.95, top=0.9, bottom=0.08)

    lead_positions = [(0,0),(0,1),(0,2), (1,0),(1,1),(1,2),
                      (2,0),(2,1),(2,2), (3,0),(3,1),(3,2)]

    for i,(r,c) in enumerate(lead_positions):
        ax = fig.add_subplot(gs[r,c])
        create_ecg_grid(ax, duration_s, y_low, y_high,
                        paper_speed_mm_s=paper_speed,
                        gain_mm_mV=amplitude_scale)
        ax.plot(t, ecg_data[i], color='black', lw=0.9, zorder=2)
        ax.set_xlim(0, duration_s)
        ax.set_ylim(y_low, y_high)
        ax.text(0.02, 0.93, lead_names[i], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, lw=0))
        ax.set_xticks([])
        ax.set_yticks([])
        if lead_names[i] in ('I','II','V1'):
            draw_calibration_pulse(ax)

    ax_r = fig.add_subplot(gs[4,:])
    n_samples_r = min(int(rhythm_duration_s * sampling_rate), n_samples)
    t_r = np.linspace(0, n_samples_r / float(sampling_rate), n_samples_r)
    strip = ecg_data[rhythm_lead_idx,:n_samples_r]

    mn_r, mx_r = np.nanmin(strip), np.nanmax(strip)
    pad_r = 0.1 * (mx_r - mn_r + 1e-12)
    y_low_r, y_high_r = mn_r - pad_r, mx_r + pad_r

    create_ecg_grid(ax_r, t_r[-1], y_low_r, y_high_r,
                    paper_speed_mm_s=paper_speed,
                    gain_mm_mV=amplitude_scale)
    ax_r.plot(t_r, strip, color='black', lw=1.0, zorder=2)
    ax_r.set_xlim(0, t_r[-1])
    ax_r.set_ylim(y_low_r, y_high_r)
    annotate_seconds(ax_r, t_r[-1], every_s=1.0)
    draw_calibration_pulse(ax_r)
    add_six_second_bracket(ax_r)
    ax_r.set_xlabel('Time (s)')
    ax_r.set_ylabel('Amplitude (mV)')
    ax_r.set_title(f'Lead {lead_names[rhythm_lead_idx]} — Rhythm Strip',
                   fontsize=12, fontweight='bold')

    tech = (f"Speed: {paper_speed} mm/s | Gain: {amplitude_scale} mm/mV | "
            f"Duration: {duration_s:.1f} s | Fs: {sampling_rate} Hz")
    fig.text(0.5, 0.04, tech, ha='center', fontsize=10, style='italic')

    if save_path:
        plt.savefig(save_path, dpi=72, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)  # Always close figure to free memory
        from PIL import Image
        img = Image.open(save_path)
        img.thumbnail((700, 700), Image.Resampling.LANCZOS)
        img.save(save_path)
    else:
        plt.show()

    return fig

# =====================
# Parallel processing functions
# =====================

def process_single_ecg(row_data, ecg_data_dir, output_dir, sampling_rate=500):
    """Process a single ECG file - designed to be called by multiprocessing"""
    deid_filename = None
    try:
        idx, row = row_data
        deid_filename = row['deid_filename']
        ecg_file_path = os.path.join(ecg_data_dir, f"{deid_filename}.npy")
        output_file_path = os.path.join(output_dir, f"{deid_filename}.png")

        # Skip if already exists
        if os.path.exists(output_file_path):
            return 'skipped', deid_filename, None

        # Check if input file exists
        if not os.path.exists(ecg_file_path):
            return 'error', deid_filename, "File not found"

        # Load and validate ECG data
        ecg_data = np.load(ecg_file_path)
        if ecg_data.ndim != 2 or ecg_data.shape[0] != 12:
            return 'error', deid_filename, f"Invalid shape: {ecg_data.shape}"

        # Generate the plot
        patient_info = {'ID': deid_filename}
        plot_hospital_ecg(
            ecg_data=ecg_data,
            sampling_rate=sampling_rate,
            paper_speed=25,
            amplitude_scale=10,
            patient_info=patient_info,
            save_path=output_file_path,
            rhythm_lead_idx=1,
            rhythm_duration_s=10,
        )
        
        return 'success', deid_filename, None

    except Exception as e:
        return 'error', deid_filename, str(e)

def process_ecg_batch_parallel(df_ecg, ecg_data_dir, output_dir, sampling_rate=500, n_processes=50):
    """
    Process ECG files in parallel using multiprocessing
    
    Args:
        df_ecg: DataFrame with ECG filenames
        ecg_data_dir: Directory containing .npy ECG files
        output_dir: Directory to save PNG images
        sampling_rate: ECG sampling rate (default: 500)
        n_processes: Number of parallel processes (default: 50)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(df_ecg)} ECG files with {n_processes} parallel processes...")
    print(f"Input directory: {ecg_data_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_ecg,
        ecg_data_dir=ecg_data_dir,
        output_dir=output_dir,
        sampling_rate=sampling_rate
    )
    
    # Convert DataFrame to list of (index, row) tuples for multiprocessing
    df_items = list(df_ecg.iterrows())
    
    # Process in parallel
    results = []
    with Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        for result in tqdm(pool.imap(process_func, df_items), 
                          total=len(df_items), 
                          desc="Processing ECGs"):
            results.append(result)
    
    # Analyze results
    processed_count = sum(1 for r in results if r[0] == 'success')
    skipped_count = sum(1 for r in results if r[0] == 'skipped')
    error_count = sum(1 for r in results if r[0] == 'error')
    error_files = [f"{r[1]} - {r[2]}" for r in results if r[0] == 'error' and r[2]]
    
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files in dataframe: {len(df_ecg)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    
    if error_files:
        print("\nError details:")
        for error in error_files[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more errors")

    return processed_count, error_count, error_files

# =====================
# Original serial processing function (for comparison)
# =====================

def process_ecg_batch(df_ecg, ecg_data_dir, output_dir, sampling_rate=500):
    """Original serial processing function"""
    os.makedirs(output_dir, exist_ok=True)
    processed_count, error_count, error_files = 0, 0, []

    print(f"Processing {len(df_ecg)} ECG files...")
    print(f"Input directory: {ecg_data_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    for idx, row in tqdm(df_ecg.iterrows(), total=len(df_ecg), desc="Processing ECGs"):
        deid_filename = None
        try:
            deid_filename = row['deid_filename']
            ecg_file_path = os.path.join(ecg_data_dir, f"{deid_filename}.npy")
            output_file_path = os.path.join(output_dir, f"{deid_filename}.png")

            if os.path.exists(output_file_path):
                processed_count += 1
                continue

            if not os.path.exists(ecg_file_path):
                error_files.append(f"{deid_filename} - File not found")
                error_count += 1
                continue

            ecg_data = np.load(ecg_file_path)
            if ecg_data.ndim != 2 or ecg_data.shape[0] != 12:
                error_files.append(f"{deid_filename} - Invalid shape: {ecg_data.shape}")
                error_count += 1
                continue

            patient_info = { 'ID': deid_filename }
            plot_hospital_ecg(
                ecg_data=ecg_data,
                sampling_rate=sampling_rate,
                paper_speed=25,
                amplitude_scale=10,
                patient_info=patient_info,
                save_path=output_file_path,
                rhythm_lead_idx=1,
                rhythm_duration_s=10,
            )
            processed_count += 1
        except Exception as e:
            error_count += 1
            error_files.append(f"{deid_filename} - Error: {str(e)}")

    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files in dataframe: {len(df_ecg)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    if error_files:
        print("\nError details:")
        for error in error_files[:10]:
            print(f"  - {error}")

    return processed_count, error_count, error_files

if __name__ == "__main__":
    ECG_DATA_DIR = "/oak/stanford/groups/euan/projects/ECGs/ECGs/processed/"
    OUTPUT_DIR   = "/oak/stanford/groups/euan/users/masadi/ecg_imgs_full"
    SAMPLING_RATE = 500
    N_PROCESSES = 120  # Number of parallel processes

    df = pd.read_csv('/oak/stanford/groups/euan/users/masadi/ecg_diagnoses.csv')
    ecg_file_list = os.listdir(ECG_DATA_DIR)
    ecg_name_list = [i.split('.')[0] for i in ecg_file_list]
    df_ecg = df[df['deid_filename'].isin(ecg_name_list)]

    # Use parallel processing
    processed, errors, error_list = process_ecg_batch_parallel(
        df_ecg=df_ecg,
        ecg_data_dir=ECG_DATA_DIR,
        output_dir=OUTPUT_DIR,
        sampling_rate=SAMPLING_RATE,
        n_processes=N_PROCESSES,
    )

    print("Parallel batch processing complete!")
    
    # Uncomment below to use original serial processing for comparison
    # processed, errors, error_list = process_ecg_batch(
    #     df_ecg=df_ecg,
    #     ecg_data_dir=ECG_DATA_DIR,
    #     output_dir=OUTPUT_DIR,
    #     sampling_rate=SAMPLING_RATE,
    # )
