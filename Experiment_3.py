"""
Experiment_3.py

Experiment 3:
Load-disturbance analysis for the battery-assisted
dual-input DC/DC converter.

Fixed condition:
    Irradiance = 1000 W/m^2
    PV available MPP = approximately 300 W

Load scenarios at t = 1 s:
    65 -> 75 ohm   : lighter load
    65 -> 55 ohm   : moderate heavier load
    65 -> 45 ohm   : stronger heavier load

Objectives:
- Verify that PV remains near its MPP during load changes.
- Evaluate battery power compensation.
- Evaluate DC-bus voltage regulation.
- Measure overshoot, undershoot, settling time,
  steady-state error, and transient power imbalance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SYSTEM PARAMETERS
# ============================================================

VREF = 400.0
VBAT = 50.0

L1 = 1e-3
CPV = 100e-6
COUT = 20e-6

IRRADIANCE = 1000.0
PV_MPP = 300.0

RLOAD_INITIAL = 65.0

T_SIM = 2.0
DT = 1e-5
STEP_TIME = 1.0


# ============================================================
# 2. CONTROLLER PARAMETERS
# ============================================================

# Keep exactly the same baseline controller as Experiment 2.

KP_V = 0.8
KI_V = 80.0

TAU_MPPT = 0.025
TAU_BAT = 0.02


# ============================================================
# 3. LOAD SCENARIOS
# ============================================================

LOAD_SCENARIOS = [
    (65.0, 75.0),   # lighter load
    (65.0, 55.0),   # moderate heavier load
    (65.0, 45.0)    # stronger heavier load
]


# ============================================================
# 4. LOAD PROFILE
# ============================================================

def load_profile(t, r_initial, r_final):

    if t < STEP_TIME:
        return r_initial

    return r_final


# ============================================================
# 5. RUN ONE LOAD-DISTURBANCE SCENARIO
# ============================================================

def run_scenario(r_initial, r_final):

    time = np.arange(0.0, T_SIM + DT, DT)

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

    duty_pv = np.zeros(n)
    duty_bat = np.zeros(n)

    power_balance_error = np.zeros(n)

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    rload[0] = r_initial

    p_mpp[0] = PV_MPP
    p_pv[0] = PV_MPP

    vout[0] = VREF

    pout[0] = VREF**2 / r_initial

    # Battery supplies the difference between load power
    # and PV power at the initial steady state.

    p_bat[0] = pout[0] - p_pv[0]

    i_bat[0] = p_bat[0] / VBAT

    duty_pv[0] = 0.85
    duty_bat[0] = 0.10

    integral_error = 0.0

    # --------------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------------

    for k in range(1, n):

        t = time[k]

        # ====================================================
        # LOAD DISTURBANCE
        # ====================================================

        rload[k] = load_profile(
            t,
            r_initial,
            r_final
        )

        # ====================================================
        # PV / MPPT
        # ====================================================

        # Irradiance remains fixed at 1000 W/m^2.
        # Therefore available MPP remains 300 W.

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
            vout[k - 1] ** 2
        ) / rload[k]

        # ====================================================
        # DC-BUS VOLTAGE CONTROLLER
        # ====================================================

        voltage_error = (
            VREF -
            vout[k - 1]
        )

        integral_error += (
            voltage_error * DT
        )

        correction_power = (
            KP_V * voltage_error +
            KI_V * integral_error
        )

        # Battery power required for balancing.

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
        # REPRESENTATIVE DUTY COMMANDS
        # ====================================================

        # PV remains near its nominal operating condition
        # because irradiance is unchanged.

        pv_fraction = (
            p_pv[k] /
            PV_MPP
        )

        duty_pv[k] = np.clip(
            0.85 * pv_fraction,
            0.0,
            0.95
        )

        # Battery duty indicator changes according to
        # required battery contribution.

        battery_fraction = (
            p_bat[k] /
            max(pout[k], 1.0)
        )

        duty_bat[k] = np.clip(
            0.10 +
            0.20 * battery_fraction,
            0.0,
            0.95
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
        "i_bat": i_bat,
        "vout": vout,
        "pout": pout,
        "duty_pv": duty_pv,
        "duty_bat": duty_bat,
        "power_balance_error": power_balance_error
    }


# ============================================================
# 6. SETTLING-TIME CALCULATION
# ============================================================

def calculate_settling_time(
    time,
    signal,
    reference,
    step_time,
    tolerance_percent=0.5
):
    """
    Calculate settling time after the load disturbance.

    The default band is +/-0.5% around 400 V:
        398 V <= Vout <= 402 V

    The voltage must enter the band and remain inside it
    for the rest of the simulation.
    """

    tolerance = (
        abs(reference) *
        tolerance_percent /
        100.0
    )

    post_indices = np.where(
        time >= step_time
    )[0]

    for index in post_indices:

        remaining_signal = signal[index:]

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
# 7. CALCULATE METRICS
# ============================================================

def calculate_metrics(
    results,
    r_initial,
    r_final
):

    time = results["time"]

    vout = results["vout"]

    p_pv = results["p_pv"]
    p_mpp = results["p_mpp"]

    p_bat = results["p_bat"]
    pout = results["pout"]

    # --------------------------------------------------------
    # Steady-state measurement window
    # --------------------------------------------------------

    steady_mask = (
        time >= 1.5
    )

    vout_steady = np.mean(
        vout[steady_mask]
    )

    pv_power_final = np.mean(
        p_pv[steady_mask]
    )

    mpp_final = np.mean(
        p_mpp[steady_mask]
    )

    battery_power_final = np.mean(
        p_bat[steady_mask]
    )

    load_power_final = np.mean(
        pout[steady_mask]
    )

    # --------------------------------------------------------
    # Post-disturbance voltage
    # --------------------------------------------------------

    post_mask = (
        time >= STEP_TIME
    )

    post_vout = vout[post_mask]

    vout_max = np.max(post_vout)
    vout_min = np.min(post_vout)

    # --------------------------------------------------------
    # Overshoot / undershoot
    # --------------------------------------------------------

    overshoot_percent = (
        (vout_max - VREF) /
        VREF
    ) * 100.0

    undershoot_percent = (
        (VREF - vout_min) /
        VREF
    ) * 100.0

    # --------------------------------------------------------
    # Steady-state error
    # --------------------------------------------------------

    steady_state_error = (
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
    # MPPT efficiency
    # --------------------------------------------------------

    if mpp_final > 0.0:

        mppt_efficiency = (
            pv_power_final /
            mpp_final
        ) * 100.0

    else:

        mppt_efficiency = np.nan

    # --------------------------------------------------------
    # Maximum transient power-balance error
    # --------------------------------------------------------

    max_power_balance_error = np.max(
        np.abs(
            results[
                "power_balance_error"
            ][post_mask]
        )
    )

    return {

        "R_initial_ohm":
            r_initial,

        "R_final_ohm":
            r_final,

        "Initial_load_power_W":
            VREF**2 / r_initial,

        "Final_load_power_W":
            load_power_final,

        "PV_power_final_W":
            pv_power_final,

        "Battery_power_final_W":
            battery_power_final,

        "Vout_steady_V":
            vout_steady,

        "Vout_max_V":
            vout_max,

        "Vout_min_V":
            vout_min,

        "Voltage_overshoot_percent":
            overshoot_percent,

        "Voltage_undershoot_percent":
            undershoot_percent,

        "Steady_state_error_V":
            steady_state_error,

        "Settling_time_s":
            settling_time,

        "MPPT_efficiency_percent":
            mppt_efficiency,

        "Max_power_balance_error_W":
            max_power_balance_error
    }


# ============================================================
# 8. RUN ALL LOAD SCENARIOS
# ============================================================

def run_all_experiments():

    all_metrics = []

    all_results = {}

    for r_initial, r_final in LOAD_SCENARIOS:

        print(
            f"\nRunning load scenario: "
            f"{r_initial:.0f} -> "
            f"{r_final:.0f} ohm"
        )

        results = run_scenario(
            r_initial,
            r_final
        )

        metrics = calculate_metrics(
            results,
            r_initial,
            r_final
        )

        all_metrics.append(metrics)

        scenario_name = (
            f"{int(r_initial)}_to_"
            f"{int(r_final)}_ohm"
        )

        all_results[
            scenario_name
        ] = results

    return (
        pd.DataFrame(all_metrics),
        all_results
    )


# ============================================================
# 9. COMPARISON PLOTS
# ============================================================

def plot_comparison(all_results):

    # --------------------------------------------------------
    # DC-bus voltage
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        plt.plot(
            results["time"],
            results["vout"],
            label=name.replace("_", " ")
        )

    plt.axhline(
        VREF,
        linestyle="--",
        label="400 V Reference"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("DC-Bus Voltage (V)")

    plt.title(
        "DC-Bus Response to Load Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # PV power
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        plt.plot(
            results["time"],
            results["p_pv"],
            label=name.replace("_", " ")
        )

    plt.axhline(
        PV_MPP,
        linestyle="--",
        label="Available PV MPP"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("PV Power (W)")

    plt.title(
        "PV Power During Load Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # Battery power
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        plt.plot(
            results["time"],
            results["p_bat"],
            label=name.replace("_", " ")
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Battery Power (W)")

    plt.title(
        "Battery Power Response to Load Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # Load power
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        plt.plot(
            results["time"],
            results["pout"],
            label=name.replace("_", " ")
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Load Power (W)")

    plt.title(
        "Load Power During Load Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    results_table, all_results = (
        run_all_experiments()
    )

    print("\n")
    print("=" * 100)
    print("EXPERIMENT 3 RESULTS")
    print("=" * 100)

    print(
        results_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_table.to_csv(
        "experiment_3_results.csv",
        index=False
    )

    print(
        "\nResults saved to: "
        "experiment_3_results.csv"
    )

    # --------------------------------------------------------
    # Generate comparison plots
    # --------------------------------------------------------

    plot_comparison(
        all_results
    )