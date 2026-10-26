"""
Experiment_4_Adaptive_PI.py

Experiment 4:
Fixed PI versus Adaptive PI controller comparison for the
battery-assisted dual-input DC/DC converter.

Fixed condition:
    Irradiance = 1000 W/m^2
    PV MPP = 300 W

Load scenarios at t = 1 s:
    65 -> 75 ohm
    65 -> 55 ohm
    65 -> 45 ohm

Controllers:
    1. Baseline fixed PI
    2. Adaptive PI using voltage-error-dependent gain scheduling

Outputs:
    - CSV comparison table
    - DC-bus voltage comparison plots
    - Adaptive Kp and Ki plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SYSTEM PARAMETERS
# ============================================================

VREF = 400.0
VBAT = 50.0

COUT = 20e-6

IRRADIANCE = 1000.0
PV_MPP = 300.0

T_SIM = 2.0
DT = 1e-5
STEP_TIME = 1.0

TAU_MPPT = 0.025
TAU_BAT = 0.02


# ============================================================
# 2. BASELINE PI PARAMETERS
# ============================================================

KP_BASE = 0.8
KI_BASE = 80.0


# ============================================================
# 3. ADAPTIVE PI PARAMETERS
# ============================================================

# Initial adaptive gain-scheduling parameters.
#
# These are experimental values and are NOT taken from
# the PSIM technical report.
#
# Larger voltage error produces larger Kp and Ki.

ALPHA_KP = 8.0
ALPHA_KI = 4.0

# Prevent excessive controller gains.

KP_MAX = 4.0
KI_MAX = 200.0


# ============================================================
# 4. LOAD SCENARIOS
# ============================================================

LOAD_SCENARIOS = [
    (65.0, 75.0),
    (65.0, 55.0),
    (65.0, 45.0)
]


# ============================================================
# 5. LOAD PROFILE
# ============================================================

def load_profile(t, r_initial, r_final):

    if t < STEP_TIME:
        return r_initial

    return r_final


# ============================================================
# 6. ADAPTIVE GAIN CALCULATION
# ============================================================

def calculate_adaptive_gains(voltage_error):
    """
    Calculate adaptive PI gains from the magnitude
    of the DC-bus voltage error.

    Small error:
        gains remain near baseline values.

    Large error:
        gains increase to provide stronger correction.

    Gain limits prevent excessive controller action.
    """

    normalized_error = (
        abs(voltage_error) / VREF
    )

    kp = KP_BASE * (
        1.0 +
        ALPHA_KP * normalized_error
    )

    ki = KI_BASE * (
        1.0 +
        ALPHA_KI * normalized_error
    )

    kp = np.clip(
        kp,
        KP_BASE,
        KP_MAX
    )

    ki = np.clip(
        ki,
        KI_BASE,
        KI_MAX
    )

    return kp, ki


# ============================================================
# 7. RUN ONE SCENARIO
# ============================================================

def run_scenario(
    r_initial,
    r_final,
    controller_type="baseline"
):

    time = np.arange(
        0.0,
        T_SIM + DT,
        DT
    )

    n = len(time)

    # --------------------------------------------------------
    # Storage arrays
    # --------------------------------------------------------

    rload = np.zeros(n)

    p_mpp = np.zeros(n)
    p_pv = np.zeros(n)

    p_bat = np.zeros(n)
    i_bat = np.zeros(n)

    vout = np.zeros(n)
    pout = np.zeros(n)

    kp_history = np.zeros(n)
    ki_history = np.zeros(n)

    power_balance_error = np.zeros(n)

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    rload[0] = r_initial

    p_mpp[0] = PV_MPP
    p_pv[0] = PV_MPP

    vout[0] = VREF

    pout[0] = (
        VREF**2 /
        r_initial
    )

    p_bat[0] = (
        pout[0] -
        p_pv[0]
    )

    i_bat[0] = (
        p_bat[0] /
        VBAT
    )

    kp_history[0] = KP_BASE
    ki_history[0] = KI_BASE

    integral_error = 0.0

    # --------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------

    for k in range(1, n):

        t = time[k]

        # ====================================================
        # LOAD
        # ====================================================

        rload[k] = load_profile(
            t,
            r_initial,
            r_final
        )

        # ====================================================
        # PV / MPPT
        # ====================================================

        p_mpp[k] = PV_MPP

        dpv_dt = (
            p_mpp[k] -
            p_pv[k - 1]
        ) / TAU_MPPT

        p_pv[k] = (
            p_pv[k - 1] +
            dpv_dt * DT
        )

        # ====================================================
        # LOAD POWER
        # ====================================================

        pout[k] = (
            vout[k - 1] ** 2 /
            rload[k]
        )

        # ====================================================
        # VOLTAGE ERROR
        # ====================================================

        voltage_error = (
            VREF -
            vout[k - 1]
        )

        # ====================================================
        # CONTROLLER SELECTION
        # ====================================================

        if controller_type == "adaptive":

            kp, ki = (
                calculate_adaptive_gains(
                    voltage_error
                )
            )

        else:

            kp = KP_BASE
            ki = KI_BASE

        kp_history[k] = kp
        ki_history[k] = ki

        # ====================================================
        # PI CONTROLLER
        # ====================================================

        integral_error += (
            voltage_error *
            DT
        )

        correction_power = (
            kp * voltage_error +
            ki * integral_error
        )

        # ====================================================
        # BATTERY POWER COMMAND
        # ====================================================

        p_bat_command = (
            pout[k] -
            p_pv[k] +
            correction_power
        )

        # ====================================================
        # BATTERY / CONVERTER DYNAMICS
        # ====================================================

        dpbat_dt = (
            p_bat_command -
            p_bat[k - 1]
        ) / TAU_BAT

        p_bat[k] = (
            p_bat[k - 1] +
            dpbat_dt * DT
        )

        i_bat[k] = (
            p_bat[k] /
            VBAT
        )

        # ====================================================
        # TOTAL CONVERTER POWER
        # ====================================================

        p_converter = (
            p_pv[k] +
            p_bat[k]
        )

        # ====================================================
        # OUTPUT CAPACITOR DYNAMICS
        # ====================================================

        if vout[k - 1] > 1.0:

            dv_dt = (
                p_converter -
                pout[k]
            ) / (
                COUT *
                vout[k - 1]
            )

        else:

            dv_dt = 0.0

        vout[k] = (
            vout[k - 1] +
            dv_dt * DT
        )

        # ====================================================
        # POWER-BALANCE ERROR
        # ====================================================

        power_balance_error[k] = (
            p_pv[k] +
            p_bat[k] -
            pout[k]
        )

    return {

        "time":
            time,

        "rload":
            rload,

        "p_mpp":
            p_mpp,

        "p_pv":
            p_pv,

        "p_bat":
            p_bat,

        "i_bat":
            i_bat,

        "vout":
            vout,

        "pout":
            pout,

        "kp":
            kp_history,

        "ki":
            ki_history,

        "power_balance_error":
            power_balance_error
    }


# ============================================================
# 8. SETTLING TIME
# ============================================================

def calculate_settling_time(
    time,
    signal,
    reference,
    step_time,
    tolerance_percent=0.5
):

    tolerance = (
        abs(reference) *
        tolerance_percent /
        100.0
    )

    post_indices = np.where(
        time >= step_time
    )[0]

    for index in post_indices:

        remaining_signal = (
            signal[index:]
        )

        if np.all(
            np.abs(
                remaining_signal -
                reference
            ) <= tolerance
        ):

            return (
                time[index] -
                step_time
            )

    return np.nan


# ============================================================
# 9. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    results,
    r_initial,
    r_final,
    controller
):

    time = results["time"]
    vout = results["vout"]

    p_pv = results["p_pv"]
    p_bat = results["p_bat"]
    pout = results["pout"]

    # --------------------------------------------------------
    # Steady-state region
    # --------------------------------------------------------

    steady_mask = (
        time >= 1.5
    )

    post_mask = (
        time >= STEP_TIME
    )

    vout_steady = np.mean(
        vout[steady_mask]
    )

    pv_final = np.mean(
        p_pv[steady_mask]
    )

    battery_final = np.mean(
        p_bat[steady_mask]
    )

    load_final = np.mean(
        pout[steady_mask]
    )

    # --------------------------------------------------------
    # Voltage extrema
    # --------------------------------------------------------

    vout_max = np.max(
        vout[post_mask]
    )

    vout_min = np.min(
        vout[post_mask]
    )

    overshoot = (
        (vout_max - VREF) /
        VREF *
        100.0
    )

    undershoot = (
        (VREF - vout_min) /
        VREF *
        100.0
    )

    # --------------------------------------------------------
    # Steady-state error
    # --------------------------------------------------------

    steady_error = (
        vout_steady -
        VREF
    )

    # --------------------------------------------------------
    # Settling time
    # --------------------------------------------------------

    settling_time = (
        calculate_settling_time(
            time,
            vout,
            VREF,
            STEP_TIME,
            tolerance_percent=0.5
        )
    )

    # --------------------------------------------------------
    # Maximum power-balance error
    # --------------------------------------------------------

    max_power_error = np.max(
        np.abs(
            results[
                "power_balance_error"
            ][post_mask]
        )
    )

    # --------------------------------------------------------
    # Maximum adaptive gains
    # --------------------------------------------------------

    max_kp = np.max(
        results["kp"][post_mask]
    )

    max_ki = np.max(
        results["ki"][post_mask]
    )

    return {

        "Controller":
            controller,

        "R_initial_ohm":
            r_initial,

        "R_final_ohm":
            r_final,

        "Initial_load_power_W":
            VREF**2 / r_initial,

        "Final_load_power_W":
            load_final,

        "PV_power_final_W":
            pv_final,

        "Battery_power_final_W":
            battery_final,

        "Vout_steady_V":
            vout_steady,

        "Vout_max_V":
            vout_max,

        "Vout_min_V":
            vout_min,

        "Overshoot_percent":
            overshoot,

        "Undershoot_percent":
            undershoot,

        "Steady_state_error_V":
            steady_error,

        "Settling_time_s":
            settling_time,

        "Max_power_balance_error_W":
            max_power_error,

        "Max_Kp":
            max_kp,

        "Max_Ki":
            max_ki
    }


# ============================================================
# 10. RUN BASELINE AND ADAPTIVE CONTROLLERS
# ============================================================

def run_all_experiments():

    metrics_list = []

    all_results = {}

    for r_initial, r_final in LOAD_SCENARIOS:

        # ====================================================
        # BASELINE PI
        # ====================================================

        print(
            f"\nBaseline PI: "
            f"{r_initial:.0f} -> "
            f"{r_final:.0f} ohm"
        )

        baseline = run_scenario(
            r_initial,
            r_final,
            controller_type="baseline"
        )

        baseline_metrics = (
            calculate_metrics(
                baseline,
                r_initial,
                r_final,
                "Baseline PI"
            )
        )

        metrics_list.append(
            baseline_metrics
        )

        # ====================================================
        # ADAPTIVE PI
        # ====================================================

        print(
            f"Adaptive PI: "
            f"{r_initial:.0f} -> "
            f"{r_final:.0f} ohm"
        )

        adaptive = run_scenario(
            r_initial,
            r_final,
            controller_type="adaptive"
        )

        adaptive_metrics = (
            calculate_metrics(
                adaptive,
                r_initial,
                r_final,
                "Adaptive PI"
            )
        )

        metrics_list.append(
            adaptive_metrics
        )

        scenario_name = (
            f"{int(r_initial)}_to_"
            f"{int(r_final)}"
        )

        all_results[
            scenario_name
        ] = {
            "baseline": baseline,
            "adaptive": adaptive
        }

    return (
        pd.DataFrame(metrics_list),
        all_results
    )


# ============================================================
# 11. PLOT CONTROLLER COMPARISON
# ============================================================

def plot_comparison(all_results):

    for scenario, data in all_results.items():

        baseline = data["baseline"]
        adaptive = data["adaptive"]

        # ----------------------------------------------------
        # Voltage comparison
        # ----------------------------------------------------

        plt.figure()

        plt.plot(
            baseline["time"],
            baseline["vout"],
            label="Baseline PI"
        )

        plt.plot(
            adaptive["time"],
            adaptive["vout"],
            label="Adaptive PI"
        )

        plt.axhline(
            VREF,
            linestyle="--",
            label="400 V Reference"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("DC-Bus Voltage (V)")

        plt.title(
            f"Voltage Response: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        # ----------------------------------------------------
        # Battery power comparison
        # ----------------------------------------------------

        plt.figure()

        plt.plot(
            baseline["time"],
            baseline["p_bat"],
            label="Baseline PI"
        )

        plt.plot(
            adaptive["time"],
            adaptive["p_bat"],
            label="Adaptive PI"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Battery Power (W)")

        plt.title(
            f"Battery Response: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        # ----------------------------------------------------
        # Adaptive Kp
        # ----------------------------------------------------

        plt.figure()

        plt.plot(
            adaptive["time"],
            adaptive["kp"]
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Kp")

        plt.title(
            f"Adaptive Kp: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.grid(True)
        plt.tight_layout()


        # ----------------------------------------------------
        # Adaptive Ki
        # ----------------------------------------------------

        plt.figure()

        plt.plot(
            adaptive["time"],
            adaptive["ki"]
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Ki")

        plt.title(
            f"Adaptive Ki: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.grid(True)
        plt.tight_layout()

    plt.show()


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    results_table, all_results = (
        run_all_experiments()
    )

    print("\n")
    print("=" * 120)
    print(
        "EXPERIMENT 4: "
        "BASELINE PI VS ADAPTIVE PI"
    )
    print("=" * 120)

    print(
        results_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results_table.to_csv(
        "experiment_4_adaptive_PI_results.csv",
        index=False
    )

    print(
        "\nResults saved to: "
        "experiment_4_adaptive_PI_results.csv"
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_comparison(
        all_results
    )