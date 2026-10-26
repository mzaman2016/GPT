"""
Experiment_2.py

Experiment 2:
Multi-scenario irradiance disturbance analysis for the
battery-assisted dual-input DC/DC converter.

Scenarios
---------
1. 1000 -> 800 W/m^2
2. 1000 -> 600 W/m^2
3. 1000 -> 400 W/m^2

Objectives
----------
- Evaluate DC-bus regulation under different irradiance disturbances.
- Measure PV and battery power response.
- Calculate voltage overshoot/undershoot.
- Calculate steady-state voltage error.
- Estimate settling time.
- Compare MPPT response.

Baseline parameters are based on Experiment 1 and the
PV/Battery Double-Input DC/DC Converter technical report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. SYSTEM PARAMETERS
# ============================================================

VREF = 400.0              # DC-bus reference voltage [V]
VBAT = 50.0               # Battery voltage [V]

L1 = 1e-3                 # Converter inductor [H]
CPV = 100e-6              # PV-side capacitor [F]
COUT = 20e-6              # Output capacitor [F]
RLOAD = 65.0              # Load resistance [ohm]

T_SIM = 2.0               # Simulation duration [s]
DT = 1e-5                 # Simulation time step [s]

STEP_TIME = 1.0           # Irradiance disturbance time [s]


# ============================================================
# 2. CONTROLLER PARAMETERS
# ============================================================

# Keep the same baseline controller used in Experiment 1.
# Do not tune these yet.

KP_V = 0.8
KI_V = 80.0

TAU_MPPT = 0.025          # MPPT response time constant [s]
TAU_BAT = 0.02            # Battery/converter response [s]


# ============================================================
# 3. IRRADIANCE SCENARIOS
# ============================================================

SCENARIOS = [
    (1000.0, 800.0),
    (1000.0, 600.0),
    (1000.0, 400.0)
]


# ============================================================
# 4. PV AVAILABLE MAXIMUM POWER MODEL
# ============================================================

def available_pv_power(irradiance):
    """
    Approximate available PV maximum power.

    Known points from the technical report:
        1000 W/m^2 -> approximately 300 W
         600 W/m^2 -> approximately 237 W

    Experiment 2 extends this relationship to other
    irradiance levels using a simple linear approximation.

    NOTE:
    This is an experimental approximation and is not yet
    a detailed physical PV-array model.
    """

    # Linear relation through:
    # (G = 1000, P = 300)
    # (G = 600,  P = 237)

    slope = (300.0 - 237.0) / (1000.0 - 600.0)

    p_mpp = 237.0 + slope * (irradiance - 600.0)

    return max(p_mpp, 0.0)


# ============================================================
# 5. IRRADIANCE PROFILE
# ============================================================

def irradiance_profile(t, g_initial, g_final):

    if t < STEP_TIME:
        return g_initial

    return g_final


# ============================================================
# 6. RUN ONE SCENARIO
# ============================================================

def run_scenario(g_initial, g_final):

    time = np.arange(0.0, T_SIM + DT, DT)

    n = len(time)

    # --------------------------------------------------------
    # Storage arrays
    # --------------------------------------------------------

    irradiance = np.zeros(n)

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

    irradiance[0] = g_initial

    p_mpp[0] = available_pv_power(g_initial)

    p_pv[0] = p_mpp[0]

    vout[0] = VREF

    pout[0] = VREF**2 / RLOAD

    # Battery initially supplies the difference between
    # load demand and PV generation.

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
        # Irradiance
        # ====================================================

        irradiance[k] = irradiance_profile(
            t,
            g_initial,
            g_final
        )

        # ====================================================
        # Available PV maximum power
        # ====================================================

        p_mpp[k] = available_pv_power(
            irradiance[k]
        )

        # ====================================================
        # MPPT dynamic response
        # ====================================================

        dpv_dt = (
            p_mpp[k] - p_pv[k - 1]
        ) / TAU_MPPT

        p_pv[k] = (
            p_pv[k - 1] +
            dpv_dt * DT
        )

        # ====================================================
        # Load power
        # ====================================================

        pout[k] = (
            vout[k - 1] ** 2
        ) / RLOAD

        # ====================================================
        # DC-BUS VOLTAGE CONTROLLER
        # ====================================================

        voltage_error = (
            VREF - vout[k - 1]
        )

        integral_error += (
            voltage_error * DT
        )

        correction_power = (
            KP_V * voltage_error +
            KI_V * integral_error
        )

        # Required battery balancing power

        p_bat_command = (
            pout[k] -
            p_pv[k] +
            correction_power
        )

        # ====================================================
        # BATTERY / CONVERTER DYNAMIC RESPONSE
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
            p_bat[k] / VBAT
        )

        # ====================================================
        # IDEAL CONVERTER POWER
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

        initial_mpp = available_pv_power(
            g_initial
        )

        pv_fraction = (
            p_pv[k] /
            max(initial_mpp, 1.0)
        )

        duty_pv[k] = np.clip(
            0.85 * pv_fraction,
            0.0,
            0.95
        )

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
        # POWER BALANCE ERROR
        # ====================================================

        power_balance_error[k] = (
            p_pv[k] +
            p_bat[k] -
            pout[k]
        )

    return {
        "time": time,
        "irradiance": irradiance,
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
# 7. SETTLING TIME
# ============================================================

def calculate_settling_time(
    time,
    signal,
    reference,
    step_time,
    tolerance_percent=0.5
):
    """
    Calculate settling time after the disturbance.

    Default settling band:
        +/- 1% of VREF

    The signal must enter the tolerance band and remain
    within it for the rest of the simulation.
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
# 8. CALCULATE SCENARIO METRICS
# ============================================================

def calculate_metrics(
    results,
    g_initial,
    g_final
):

    time = results["time"]

    vout = results["vout"]

    p_pv = results["p_pv"]
    p_mpp = results["p_mpp"]

    p_bat = results["p_bat"]

    # --------------------------------------------------------
    # Steady-state window
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

    # --------------------------------------------------------
    # Post-step voltage response
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
    # Steady-state voltage error
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

    if mpp_final > 0:

        mppt_efficiency = (
            pv_power_final /
            mpp_final
        ) * 100.0

    else:

        mppt_efficiency = np.nan

    # --------------------------------------------------------
    # Maximum power-balance error
    # --------------------------------------------------------

    max_power_balance_error = np.max(
        np.abs(
            results[
                "power_balance_error"
            ][post_mask]
        )
    )

    return {
        "G_initial_Wm2": g_initial,
        "G_final_Wm2": g_final,

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
# 9. RUN ALL EXPERIMENTS
# ============================================================

def run_all_experiments():

    all_metrics = []

    all_results = {}

    for g_initial, g_final in SCENARIOS:

        print(
            f"\nRunning scenario: "
            f"{g_initial:.0f} -> "
            f"{g_final:.0f} W/m^2"
        )

        results = run_scenario(
            g_initial,
            g_final
        )

        metrics = calculate_metrics(
            results,
            g_initial,
            g_final
        )

        all_metrics.append(metrics)

        scenario_name = (
            f"{int(g_initial)}_to_"
            f"{int(g_final)}"
        )

        all_results[
            scenario_name
        ] = results

    return (
        pd.DataFrame(all_metrics),
        all_results
    )


# ============================================================
# 10. PLOT COMPARISON
# ============================================================

def plot_comparison(all_results):

    # --------------------------------------------------------
    # Output-voltage comparison
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        label = name.replace(
            "_",
            " "
        )

        plt.plot(
            results["time"],
            results["vout"],
            label=label
        )

    plt.axhline(
        VREF,
        linestyle="--",
        label="400 V Reference"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("DC-Bus Voltage (V)")

    plt.title(
        "DC-Bus Response Under "
        "Irradiance Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # PV-power comparison
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        label = name.replace(
            "_",
            " "
        )

        plt.plot(
            results["time"],
            results["p_pv"],
            label=label
        )

    plt.xlabel("Time (s)")
    plt.ylabel("PV Power (W)")

    plt.title(
        "PV Power Response Under "
        "Irradiance Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # Battery-power comparison
    # --------------------------------------------------------

    plt.figure()

    for name, results in all_results.items():

        label = name.replace(
            "_",
            " "
        )

        plt.plot(
            results["time"],
            results["p_bat"],
            label=label
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Battery Power (W)")

    plt.title(
        "Battery Power Response Under "
        "Irradiance Disturbances"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    results_table, all_results = (
        run_all_experiments()
    )

    print("\n")
    print("=" * 80)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 80)

    print(
        results_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save experimental results
    # --------------------------------------------------------

    results_table.to_csv(
        "experiment_2_results.csv",
        index=False
    )

    print(
        "\nResults saved to: "
        "experiment_2_results.csv"
    )

    # --------------------------------------------------------
    # Generate comparison plots
    # --------------------------------------------------------

    plot_comparison(
        all_results
    )