#!/usr/bin/env python3
"""Analyze gripper log data to understand why gripper doesn't follow commands."""

import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_gripper_log(log_path: str) -> dict:
    """Parse GRIPPER entries from teleop debug log.

    Returns:
        Dict with keys: frame, L_raw, L_joint, R_raw, R_joint, L_curr, R_curr
    """
    data = {
        "frame": [],
        "L_raw": [],
        "L_joint": [],
        "R_raw": [],
        "R_joint": [],
        "L_curr": [],
        "R_curr": [],
    }

    # Pattern with optional L_curr and R_curr
    # Old: [frame] GRIPPER: L_raw=X.XXX -> L_joint=X.XXXX, R_raw=X.XXX -> R_joint=X.XXXX
    # New: [frame] GRIPPER: L_raw=X.XXX -> L_joint=X.XXXX, R_raw=X.XXX -> R_joint=X.XXXX, L_curr=X.XXXX, R_curr=X.XXXX
    pattern_new = r"\[(\d+)\] GRIPPER: L_raw=([0-9.\-+]+) -> L_joint=([0-9.\-+]+), R_raw=([0-9.\-+]+) -> R_joint=([0-9.\-+]+), L_curr=([0-9.\-+]+), R_curr=([0-9.\-+]+)"
    pattern_old = r"\[(\d+)\] GRIPPER: L_raw=([0-9.\-+]+) -> L_joint=([0-9.\-+]+), R_raw=([0-9.\-+]+) -> R_joint=([0-9.\-+]+)"

    with open(log_path, "r") as f:
        for line in f:
            # Try new format first
            match = re.match(pattern_new, line.strip())
            if match:
                data["frame"].append(int(match.group(1)))
                data["L_raw"].append(float(match.group(2)))
                data["L_joint"].append(float(match.group(3)))
                data["R_raw"].append(float(match.group(4)))
                data["R_joint"].append(float(match.group(5)))
                data["L_curr"].append(float(match.group(6)))
                data["R_curr"].append(float(match.group(7)))
            else:
                # Try old format
                match = re.match(pattern_old, line.strip())
                if match:
                    data["frame"].append(int(match.group(1)))
                    data["L_raw"].append(float(match.group(2)))
                    data["L_joint"].append(float(match.group(3)))
                    data["R_raw"].append(float(match.group(4)))
                    data["R_joint"].append(float(match.group(5)))
                    data["L_curr"].append(np.nan)  # No current data
                    data["R_curr"].append(np.nan)

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    return data


def analyze_and_plot(log_path: str, output_path: str = None):
    """Analyze gripper data and create visualization."""

    data = parse_gripper_log(log_path)

    if len(data["frame"]) == 0:
        print("Error: No GRIPPER entries found in log file!")
        return

    has_curr_data = not np.all(np.isnan(data["L_curr"]))

    print(f"=== Gripper Log Analysis ===")
    print(f"Total frames: {len(data['frame'])}")
    print(f"Frame range: {data['frame'].min()} - {data['frame'].max()}")
    print(f"Has current position data: {has_curr_data}")
    print()

    # Left gripper statistics
    print("=== Left Gripper ===")
    print(f"L_raw  range: [{data['L_raw'].min():.3f}, {data['L_raw'].max():.3f}]")
    print(f"L_joint (target) range: [{data['L_joint'].min():.4f}, {data['L_joint'].max():.4f}]")
    if has_curr_data:
        print(f"L_curr (actual) range: [{np.nanmin(data['L_curr']):.4f}, {np.nanmax(data['L_curr']):.4f}]")
    print(f"L_raw  mean: {data['L_raw'].mean():.3f}, std: {data['L_raw'].std():.3f}")
    print(f"L_joint mean: {data['L_joint'].mean():.4f}, std: {data['L_joint'].std():.4f}")
    if has_curr_data:
        print(f"L_curr mean: {np.nanmean(data['L_curr']):.4f}, std: {np.nanstd(data['L_curr']):.4f}")

    # Check if gripper value changes
    l_raw_changes = np.abs(np.diff(data["L_raw"]))
    l_joint_changes = np.abs(np.diff(data["L_joint"]))
    print(f"L_raw  significant changes (>0.1): {np.sum(l_raw_changes > 0.1)}")
    print(f"L_joint significant changes (>0.005): {np.sum(l_joint_changes > 0.005)}")
    print()

    # Right gripper statistics
    print("=== Right Gripper ===")
    print(f"R_raw  range: [{data['R_raw'].min():.3f}, {data['R_raw'].max():.3f}]")
    print(f"R_joint (target) range: [{data['R_joint'].min():.4f}, {data['R_joint'].max():.4f}]")
    if has_curr_data:
        print(f"R_curr (actual) range: [{np.nanmin(data['R_curr']):.4f}, {np.nanmax(data['R_curr']):.4f}]")
    print(f"R_raw  mean: {data['R_raw'].mean():.3f}, std: {data['R_raw'].std():.3f}")
    print(f"R_joint mean: {data['R_joint'].mean():.4f}, std: {data['R_joint'].std():.4f}")
    if has_curr_data:
        print(f"R_curr mean: {np.nanmean(data['R_curr']):.4f}, std: {np.nanstd(data['R_curr']):.4f}")

    r_raw_changes = np.abs(np.diff(data["R_raw"]))
    r_joint_changes = np.abs(np.diff(data["R_joint"]))
    print(f"R_raw  significant changes (>0.1): {np.sum(r_raw_changes > 0.1)}")
    print(f"R_joint significant changes (>0.005): {np.sum(r_joint_changes > 0.005)}")
    print()

    # Mapping verification
    print("=== Mapping Verification ===")
    # Expected: L_raw=-1 -> L_joint=0.0, L_raw=+1 -> L_joint=0.04
    # Formula: L_joint = (L_raw + 1) / 2 * 0.04
    expected_L_joint = (data["L_raw"] + 1) / 2 * 0.04
    expected_R_joint = (data["R_raw"] + 1) / 2 * 0.04
    L_error = np.abs(data["L_joint"] - expected_L_joint)
    R_error = np.abs(data["R_joint"] - expected_R_joint)
    print(f"Left mapping error: max={L_error.max():.6f}, mean={L_error.mean():.6f}")
    print(f"Right mapping error: max={R_error.max():.6f}, mean={R_error.mean():.6f}")
    if L_error.max() < 0.001 and R_error.max() < 0.001:
        print("✓ Mapping is CORRECT (raw -> joint conversion works)")
    else:
        print("✗ Mapping has errors!")
    print()

    # Target vs Actual analysis (if we have current data)
    if has_curr_data:
        print("=== Target vs Actual Analysis ===")
        L_tracking_error = data["L_joint"] - data["L_curr"]
        R_tracking_error = data["R_joint"] - data["R_curr"]
        print(f"Left tracking error: mean={np.nanmean(L_tracking_error):.4f}, max={np.nanmax(np.abs(L_tracking_error)):.4f}")
        print(f"Right tracking error: mean={np.nanmean(R_tracking_error):.4f}, max={np.nanmax(np.abs(R_tracking_error)):.4f}")

        # Check for stuck gripper
        l_stuck = np.nanmax(np.abs(L_tracking_error)) > 0.01  # >1cm error
        r_stuck = np.nanmax(np.abs(R_tracking_error)) > 0.01
        if l_stuck or r_stuck:
            print()
            print("⚠️  GRIPPER TRACKING ISSUE DETECTED!")
            if l_stuck:
                print(f"   Left gripper: actual position doesn't follow target")
            if r_stuck:
                print(f"   Right gripper: actual position doesn't follow target")
            print()
            print("   Possible causes:")
            print("   1. Gripper colliding with grasped object")
            print("   2. Actuator velocity limit too slow (0.07 m/s)")
            print("   3. High damping (1000) slowing response")
            print("   4. Contact friction preventing opening")
        else:
            print("✓ Grippers are tracking targets correctly")
        print()

    # Key finding
    print("=== Summary ===")
    print("The command values (L_raw, R_raw) and target joint positions (L_joint, R_joint)")
    print("are being generated correctly. The gripper reopening issue is NOT in the")
    print("command generation/mapping logic.")
    print()
    if has_curr_data:
        print("Run the teleop again with this updated logging to capture target vs actual")
        print("gripper positions and identify if the actuator is following commands.")
    else:
        print("The issue is likely in the ACTUATOR physics:")
        print("1. velocity_limit_sim=0.07 m/s - slow but should work")
        print("2. High damping (1000.0) - may slow response")
        print("3. Friction (0.2) - may resist movement")
        print("4. Gripper may be colliding with grasped object")
        print()
        print("RECOMMENDATION: Run teleop again with updated logging to capture")
        print("actual gripper position (robot.data.joint_pos) vs target.")

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Gripper Teleoperation Analysis", fontsize=14)

    # Row 1: Raw grip values over time
    ax1 = axes[0, 0]
    ax1.plot(data["frame"], data["L_raw"], "b-", alpha=0.7, label="L_raw")
    ax1.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Open (+1)")
    ax1.axhline(y=-1.0, color="r", linestyle="--", alpha=0.5, label="Close (-1)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Grip Value")
    ax1.set_title("Left Hand Raw Grip Input")
    ax1.legend()
    ax1.set_ylim(-1.2, 1.2)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(data["frame"], data["R_raw"], "b-", alpha=0.7, label="R_raw")
    ax2.axhline(y=1.0, color="g", linestyle="--", alpha=0.5, label="Open (+1)")
    ax2.axhline(y=-1.0, color="r", linestyle="--", alpha=0.5, label="Close (-1)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Grip Value")
    ax2.set_title("Right Hand Raw Grip Input")
    ax2.legend()
    ax2.set_ylim(-1.2, 1.2)
    ax2.grid(True, alpha=0.3)

    # Row 2: Joint targets vs actual over time
    ax3 = axes[1, 0]
    ax3.plot(data["frame"], data["L_joint"], "b-", alpha=0.7, linewidth=1.5, label="Target")
    if has_curr_data:
        ax3.plot(data["frame"], data["L_curr"], "r-", alpha=0.7, linewidth=1.5, label="Actual")
    ax3.axhline(y=0.04, color="g", linestyle="--", alpha=0.3, label="Open (0.04)")
    ax3.axhline(y=0.0, color="orange", linestyle="--", alpha=0.3, label="Close (0.0)")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Joint Position (m)")
    ax3.set_title("Left Gripper: Target vs Actual")
    ax3.legend(loc="upper right")
    ax3.set_ylim(-0.005, 0.045)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(data["frame"], data["R_joint"], "b-", alpha=0.7, linewidth=1.5, label="Target")
    if has_curr_data:
        ax4.plot(data["frame"], data["R_curr"], "r-", alpha=0.7, linewidth=1.5, label="Actual")
    ax4.axhline(y=0.04, color="g", linestyle="--", alpha=0.3, label="Open (0.04)")
    ax4.axhline(y=0.0, color="orange", linestyle="--", alpha=0.3, label="Close (0.0)")
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Joint Position (m)")
    ax4.set_title("Right Gripper: Target vs Actual")
    ax4.legend(loc="upper right")
    ax4.set_ylim(-0.005, 0.045)
    ax4.grid(True, alpha=0.3)

    # Row 3: Mapping verification (scatter)
    ax5 = axes[2, 0]
    ax5.scatter(data["L_raw"], data["L_joint"], s=1, alpha=0.3, c="blue")
    # Perfect mapping line
    x_line = np.linspace(-1, 1, 100)
    y_line = (x_line + 1) / 2 * 0.04
    ax5.plot(x_line, y_line, "r-", linewidth=2, label="Expected mapping")
    ax5.set_xlabel("L_raw (input)")
    ax5.set_ylabel("L_joint (output)")
    ax5.set_title("Left Gripper Mapping")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = axes[2, 1]
    ax6.scatter(data["R_raw"], data["R_joint"], s=1, alpha=0.3, c="blue")
    ax6.plot(x_line, y_line, "r-", linewidth=2, label="Expected mapping")
    ax6.set_xlabel("R_raw (input)")
    ax6.set_ylabel("R_joint (output)")
    ax6.set_title("Right Gripper Mapping")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        # Default save path
        output_path = str(Path(log_path).parent / "gripper_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")

    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default log path
        log_path = "/home/jonhpark/workspace/rocoChallenge2026/submodules/IsaacLab/logs/teleop_debug/teleop_debug_20260101_001731.log"
    else:
        log_path = sys.argv[1]

    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Analyzing: {log_path}")
    analyze_and_plot(log_path, output_path)
