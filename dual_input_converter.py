"""
dual_input_converter.py

Experiment 1:
Ideal battery-assisted dual-input DC/DC converter.

Objectives
----------
1. PV operates at/near maximum power point.
2. Battery balances the difference between PV and load power.
3. DC-bus voltage is regulated at 400 V.
4. Irradiance changes from 1000 W/m^2 to 600 W/m^2 at t = 1 s.

Initial parameters are based on the PSIM technical report.
"""

import numpy as np
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

IRRADIANCE_INITIAL = 1000.0
IRRADIANCE_FINAL = 600.0
IRRADIANCE_STEP_TIME = 1.0

PMPP_INITIAL = 300.0       # Approx. PV MPP at 1000 W/m^2
PMPP_FINAL = 237.0         # Approx. PV MPP at 600 W/m^2

# Battery/converter response time constant.
# Prevents the battery from responding instantaneously to
# changes in the required balancing power.
TAU_BAT = 0.02


# ============================================================
# 2. CONTROLLER PARAMETERS
# ============================================================

# Outer DC-bus voltage PI controller.
# These are initial simulation gains, not values reported
# in the technical report. They can be tuned experimentally.

KP_V = 0.8
KI_V = 80.0

# First-order representation of PV MPPT dynamics.
# This lets the PV power move toward the available MPP
# rather than changing instantaneously.

TAU_MPPT = 0.025           # MPPT response time constant [s]


# ============================================================
# 3. IRRADIANCE PROFILE
# ============================================================

def irradiance_profile(t):
    """Return irradiance at simulation time t."""

    if t < IRRADIANCE_STEP_TIME:
        return IRRADIANCE_INITIAL

    return IRRADIANCE_FINAL


# ============================================================
# 4. PV MAXIMUM AVAILABLE POWER
# ============================================================

def available_pv_power(irradiance):
    """
    Approximate maximum available PV power.

    The technical report gives:
        1000 W/m^2 -> approximately 300 W
         600 W/m^2 -> approximately 237 W

    For Experiment 1 these two operating points are used
    directly so that the Python experiment can be compared
    with the reported PSIM case.
    """

    if irradiance >= IRRADIANCE_INITIAL:
        return PMPP_INITIAL

    if irradiance <= IRRADIANCE_FINAL:
        return PMPP_FINAL

    # Linear interpolation between the two report values
    fraction = (
        (irradiance - IRRADIANCE_FINAL) /
        (IRRADIANCE_INITIAL - IRRADIANCE_FINAL)
    )

    return PMPP_FINAL + fraction * (
        PMPP_INITIAL - PMPP_FINAL
    )


# ============================================================
# 5. SIMULATION
# ============================================================

def run_simulation():

    time = np.arange(0.0, T_SIM + DT, DT)
    n = len(time)

    # --------------------------------------------------------
    # Storage
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

    # Start at the desired DC-bus voltage.
    vout[0] = VREF

    # Initial PV operating point
    p_pv[0] = PMPP_INITIAL
    p_mpp[0] = PMPP_INITIAL

    # Initial irradiance
    irradiance[0] = IRRADIANCE_INITIAL

    # Initial load power
    pout[0] = VREF**2 / RLOAD

    # Initialize battery at the power required to maintain
    # steady-state power balance.
    p_bat[0] = pout[0] - p_pv[0]

    # Initial battery current
    i_bat[0] = p_bat[0] / VBAT

    integral_error = 0.0

    # Nominal duty commands from technical report
    duty_pv[0] = 0.85
    duty_bat[0] = 0.10

    # --------------------------------------------------------
    # Main simulation loop
    # --------------------------------------------------------

    for k in range(1, n):

        t = time[k]

        # ====================================================
        # Irradiance and PV MPPT
        # ====================================================

        irradiance[k] = irradiance_profile(t)

        p_mpp[k] = available_pv_power(irradiance[k])

        # First-order MPPT response
        dpv_dt = (
            p_mpp[k] - p_pv[k - 1]
        ) / TAU_MPPT

        p_pv[k] = p_pv[k - 1] + dpv_dt * DT

        # ====================================================
        # Load
        # ====================================================

        pout[k] = vout[k - 1] ** 2 / RLOAD

        # ====================================================
        # DC-BUS VOLTAGE CONTROLLER
        # ====================================================

        voltage_error = VREF - vout[k - 1]

        integral_error += voltage_error * DT

        correction_power = (
            KP_V * voltage_error +
            KI_V * integral_error
        )

        # Required battery power:
        # load requirement - PV contribution
        # plus voltage-regulation correction

        p_bat_command = (
            pout[k] -
            p_pv[k] +
            correction_power
        )

        # ----------------------------------------------------
        # Battery/converter dynamic response
        # ----------------------------------------------------
        # The battery cannot instantaneously reach the commanded
        # balancing power. A first-order response is used to
        # represent the combined battery/converter dynamics.

        dpbat_dt = (
            p_bat_command - p_bat[k - 1]
        ) / TAU_BAT

        p_bat[k] = (
            p_bat[k - 1] +
            dpbat_dt * DT
        )

        # Battery current
        i_bat[k] = p_bat[k] / VBAT

        # ====================================================
        # IDEAL CONVERTER POWER BALANCE
        # ====================================================

        p_converter = p_pv[k] + p_bat[k]

        # Capacitor energy relation:
        #
        # E = 0.5 C V^2
        #
        # dE/dt = Pin - Pload
        #
        # C V dV/dt = Pin - Pload

        if vout[k - 1] > 1.0:

            dv_dt = (
                p_converter - pout[k]
            ) / (
                COUT * vout[k - 1]
            )

        else:
            dv_dt = 0.0

        vout[k] = vout[k - 1] + dv_dt * DT

        # ====================================================
        # REPRESENTATIVE DUTY COMMANDS
        # ====================================================

        # For Experiment 1 these are control indicators,
        # not a switching-level converter model.

        pv_fraction = p_pv[k] / PMPP_INITIAL

        duty_pv[k] = np.clip(
            0.85 * pv_fraction,
            0.0,
            0.95
        )

        battery_fraction = p_bat[k] / max(pout[k], 1.0)

        duty_bat[k] = np.clip(
            0.10 + 0.20 * battery_fraction,
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
# 6. PERFORMANCE METRICS
# ============================================================

def calculate_metrics(results):

    time = results["time"]
    vout = results["vout"]
    p_pv = results["p_pv"]
    p_mpp = results["p_mpp"]

    # Evaluate after irradiance transient
    mask = time >= 1.2

    steady_vout = np.mean(vout[mask])
    steady_pv = np.mean(p_pv[mask])
    steady_mpp = np.mean(p_mpp[mask])

    voltage_error = steady_vout - VREF

    mppt_efficiency = (
        steady_pv / steady_mpp
    ) * 100.0

    post_step = time >= IRRADIANCE_STEP_TIME

    max_voltage = np.max(vout[post_step])
    min_voltage = np.min(vout[post_step])

    print("\n======================================")
    print("EXPERIMENT 1 RESULTS")
    print("======================================")

    print(f"Reference voltage      : {VREF:.2f} V")
    print(f"Steady-state Vout      : {steady_vout:.2f} V")
    print(f"Voltage error          : {voltage_error:.4f} V")

    print()

    print(f"Final PV power         : {steady_pv:.2f} W")
    print(f"Final available MPP    : {steady_mpp:.2f} W")
    print(f"MPPT efficiency        : {mppt_efficiency:.2f} %")

    print()

    print(f"Post-step max Vout     : {max_voltage:.2f} V")
    print(f"Post-step min Vout     : {min_voltage:.2f} V")


# ============================================================
# 7. PLOTS
# ============================================================

def plot_results(results):

    time = results["time"]

    # --------------------------------------------------------
    # Irradiance
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        results["irradiance"]
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Irradiance (W/m²)")
    plt.title("PV Irradiance Profile")

    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # PV power / MPP
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        results["p_mpp"],
        label="Available MPP"
    )

    plt.plot(
        time,
        results["p_pv"],
        label="Tracked PV Power"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("PV Maximum Power Point Tracking")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # Output voltage
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        results["vout"],
        label="Vout"
    )

    plt.axhline(
        VREF,
        linestyle="--",
        label="400 V Reference"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Output Voltage (V)")
    plt.title("DC-Bus Voltage Regulation")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # Power sharing
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        results["p_pv"],
        label="PV Power"
    )

    plt.plot(
        time,
        results["p_bat"],
        label="Battery Power"
    )

    plt.plot(
        time,
        results["pout"],
        label="Load Power"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("PV-Battery Power Sharing")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --------------------------------------------------------
    # Duty commands
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        time,
        results["duty_pv"],
        label="PV Duty Command"
    )

    plt.plot(
        time,
        results["duty_bat"],
        label="Battery Duty Command"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Duty Ratio")
    plt.title("Dual-Input Converter Control Commands")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plt.show()


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    results = run_simulation()

    calculate_metrics(results)

    plot_results(results)