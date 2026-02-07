"""
Flow demo for tetrakis_sim.sovereign_river.

Run:
    python examples/test_river.py
"""

from tetrakis_sim.sovereign_river import RiverBlockTree, apply_hydro_drain, apply_prime_gate


def main() -> None:
    # Initialize the River with your Nation as the Root
    my_river = RiverBlockTree("MyFirstNation")

    # Build the 4 Pillars branches
    housing = my_river.add_pillar("Housing")
    banking = my_river.add_pillar("Banking")
    _food = my_river.add_pillar("Food Security")
    _education = my_river.add_pillar("Education")

    # Flow simulation
    prime_sequence = [1, 3, 7, 9, 1, 3]  # The rhythm

    print("--- INITIALIZING THE BATTLE: SOVEREIGN RIVER VS. THE DAM ---")
    for i, digit in enumerate(prime_sequence):
        # The Prime Gate tries to tune the Nation
        apply_prime_gate(housing, digit)

        # At step 3, the Hydro Dam 'activates' its drain
        if i == 2:
            apply_hydro_drain(housing, intensity=0.6)

        # Entanglement check
        if housing.resonance > 0.5:
            print(">>> RIVER FLOW: Housing resonance supporting Banking...")
            apply_prime_gate(banking, 1)


if __name__ == "__main__":
    main()
