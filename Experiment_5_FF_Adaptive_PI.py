"""
Experiment_5_FF_Adaptive_PI.py

Experiment 5:
Comparison of:
    1. Baseline PI
    2. Adaptive PI
    3. Feed-Forward + Adaptive PI

for load disturbances in the battery-assisted
dual-input DC/DC converter.

Fixed condition:
    Irradiance = 1000 W/m^2
    PV MPP = 300 W

Load scenarios:
    65 -> 75 ohm
    65 -> 55 ohm
    65 -> 45 ohm

The feed-forward controller estimates the change in
required battery power caused by the load disturbance
and provides an immediate compensating contribution.

The PI/adaptive PI loop then corrects the remaining
DC-bus voltage error.
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

# Same adaptive controller used in Experiment 4.

ALPHA_KP = 8.0
ALPHA_KI = 4.0

KP_MAX = 4.0
KI_MAX = 200.0


# ============================================================
# 4. FEED-FORWARD PARAMETER
# ============================================================

# Feed-forward gain.
#
# KFF = 1.0 means that the controller attempts to compensate
# for the full estimated change in load demand.
#
# Keep this at 1.0 for the first experiment.

KFF = 1.0


# ============================================================
# 5. LOAD SCENARIOS
# ============================================================

LOAD_SCENARIOS = [
    (65.0, 75.0),
    (65.0, 55.0),
    (65.0, 45.0)
]


# ============================================================
# 6. LOAD PROFILE
# ============================================================

def load_profile(t, r_initial, r_final):

    if t < STEP_TIME:
        return r_initial

    return r_final


# ============================================================
# 7. ADAPTIVE PI GAINS
# ============================================================

def calculate_adaptive_gains(voltage_error):

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
# 8. RUN ONE SCENARIO
# ============================================================

def run_scenario(
    r_initial,
    r_final,
    controller_type
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
    p_bat_command = np.zeros(n)

    i_bat = np.zeros(n)

    vout = np.zeros(n)
    pout = np.zeros(n)

    kp_history = np.zeros(n)
    ki_history = np.zeros(n)

    feedforward_history = np.zeros(n)

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

    p_bat_command[0] = p_bat[0]

    i_bat[0] = (
        p_bat[0] /
        VBAT
    )

    kp_history[0] = KP_BASE
    ki_history[0] = KI_BASE

    integral_error = 0.0

    # Initial nominal load power.
    initial_load_power = (
        VREF**2 /
        r_initial
    )

    # --------------------------------------------------------
    # Main simulation loop
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
        # ACTUAL LOAD POWER
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
        # CONTROLLER GAINS
        # ====================================================

        if controller_type in [
            "adaptive",
            "ff_adaptive"
        ]:

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
        # PI FEEDBACK
        # ====================================================

        integral_error += (
            voltage_error * DT
        )

        pi_correction = (
            kp * voltage_error +
            ki * integral_error
        )

        # ====================================================
        # BASE BATTERY POWER REQUIREMENT
        # ====================================================

        #
        # This is the battery power required from the
        # instantaneous power-balance calculation.
        #

        base_battery_power = (
            pout[k] -
            p_pv[k]
        )

        # ====================================================
        # FEED-FORWARD COMPENSATION
        # ====================================================

        feedforward_power = 0.0

        if controller_type == "ff_adaptive":

            #
            # Estimate load power at the desired 400-V bus.
            #
            # This allows the controller to respond to the
            # load change without waiting for a large voltage
            # error to develop.
            #

            expected_load_power = (
                VREF**2 /
                rload[k]
            )

            #
            # Change in required load power relative to the
            # original operating point.
            #

            delta_load_power = (
                expected_load_power -
                initial_load_power
            )

            feedforward_power = (
                KFF *
                delta_load_power
            )

        feedforward_history[k] = (
            feedforward_power
        )

        # ====================================================
        # BATTERY COMMAND
        # ====================================================

        if controller_type == "ff_adaptive":

            #
            # For the feed-forward controller, start from the
            # ORIGINAL steady-state battery operating point.
            #
            # Feed-forward supplies the estimated change in
            # load requirement.
            #

            initial_battery_power = (
                initial_load_power -
                PV_MPP
            )

            p_bat_command[k] = (
                initial_battery_power +
                feedforward_power +
                pi_correction
            )

        else:

            #
            # Baseline and adaptive PI retain the same
            # formulation used in Experiment 4.
            #

            p_bat_command[k] = (
                base_battery_power +
                pi_correction
            )

        # ====================================================
        # BATTERY / CONVERTER DYNAMICS
        # ====================================================

        dpbat_dt = (
            p_bat_command[k] -
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

        "time": time,
        "rload": rload,

        "p_mpp": p_mpp,
        "p_pv": p_pv,

        "p_bat": p_bat,
        "p_bat_command": p_bat_command,

        "i_bat": i_bat,

        "vout": vout,
        "pout": pout,

        "kp": kp_history,
        "ki": ki_history,

        "feedforward":
            feedforward_history,

        "power_balance_error":
            power_balance_error
    }


# ============================================================
# 9. SETTLING TIME
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
# 10. PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    results,
    r_initial,
    r_final,
    controller_name
):

    time = results["time"]
    vout = results["vout"]

    p_pv = results["p_pv"]
    p_bat = results["p_bat"]
    pout = results["pout"]

    post_mask = (
        time >= STEP_TIME
    )

    steady_mask = (
        time >= 1.5
    )

    # --------------------------------------------------------
    # Steady-state values
    # --------------------------------------------------------

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
    # Power-balance error
    # --------------------------------------------------------

    max_power_error = np.max(
        np.abs(
            results[
                "power_balance_error"
            ][post_mask]
        )
    )

    # --------------------------------------------------------
    # Controller gain statistics
    # --------------------------------------------------------

    max_kp = np.max(
        results["kp"][post_mask]
    )

    max_ki = np.max(
        results["ki"][post_mask]
    )

    max_ff = np.max(
        np.abs(
            results[
                "feedforward"
            ][post_mask]
        )
    )

    return {

        "Controller":
            controller_name,

        "R_initial_ohm":
            r_initial,

        "R_final_ohm":
            r_final,

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
            max_ki,

        "Max_FF_power_W":
            max_ff
    }


# ============================================================
# 11. RUN ALL CONTROLLERS
# ============================================================

def run_all_experiments():

    metrics_list = []
    all_results = {}

    controller_configs = [

        (
            "baseline",
            "Baseline PI"
        ),

        (
            "adaptive",
            "Adaptive PI"
        ),

        (
            "ff_adaptive",
            "FF + Adaptive PI"
        )
    ]

    for r_initial, r_final in LOAD_SCENARIOS:

        scenario_name = (
            f"{int(r_initial)}_to_"
            f"{int(r_final)}"
        )

        all_results[
            scenario_name
        ] = {}

        for (
            controller_type,
            controller_name
        ) in controller_configs:

            print(
                f"\n{controller_name}: "
                f"{r_initial:.0f} -> "
                f"{r_final:.0f} ohm"
            )

            results = run_scenario(
                r_initial,
                r_final,
                controller_type
            )

            metrics = calculate_metrics(
                results,
                r_initial,
                r_final,
                controller_name
            )

            metrics_list.append(
                metrics
            )

            all_results[
                scenario_name
            ][
                controller_type
            ] = results

    return (
        pd.DataFrame(metrics_list),
        all_results
    )


# ============================================================
# 12. PLOTS
# ============================================================

def plot_comparison(all_results):

    for scenario, data in all_results.items():

        baseline = data["baseline"]
        adaptive = data["adaptive"]
        ff_adaptive = data["ff_adaptive"]

        # ----------------------------------------------------
        # DC-BUS VOLTAGE
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

        plt.plot(
            ff_adaptive["time"],
            ff_adaptive["vout"],
            label="FF + Adaptive PI"
        )

        plt.axhline(
            VREF,
            linestyle="--",
            label="400 V Reference"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("DC-Bus Voltage (V)")

        plt.title(
            f"Controller Comparison: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        # ----------------------------------------------------
        # BATTERY POWER
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

        plt.plot(
            ff_adaptive["time"],
            ff_adaptive["p_bat"],
            label="FF + Adaptive PI"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Battery Power (W)")

        plt.title(
            f"Battery Power: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()


        # ----------------------------------------------------
        # FEED-FORWARD POWER
        # ----------------------------------------------------

        plt.figure()

        plt.plot(
            ff_adaptive["time"],
            ff_adaptive["feedforward"]
        )

        plt.xlabel("Time (s)")
        plt.ylabel(
            "Feed-Forward Power (W)"
        )

        plt.title(
            f"Feed-Forward Compensation: "
            f"{scenario.replace('_', ' ')} ohm"
        )

        plt.grid(True)
        plt.tight_layout()

    plt.show()


# ============================================================
# 13. MAIN
# ============================================================

if __name__ == "__main__":

    results_table, all_results = (
        run_all_experiments()
    )

    print("\n")
    print("=" * 140)

    print(
        "EXPERIMENT 5: "
        "BASELINE PI VS ADAPTIVE PI VS "
        "FEED-FORWARD + ADAPTIVE PI"
    )

    print("=" * 140)

    print(
        results_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    results_table.to_csv(
        "experiment_5_FF_adaptive_PI_results.csv",
        index=False
    )

    print(
        "\nResults saved to: "
        "experiment_5_FF_adaptive_PI_results.csv"
    )

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    plot_comparison(
        all_results
    )