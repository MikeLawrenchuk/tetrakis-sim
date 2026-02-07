"""
Flow demo for tetrakis_sim.sovereign_river.

Run:
    python examples/test_river.py
"""

from tetrakis_sim.sovereign_river import RiverBlockTree, apply_prime_gate


def main() -> None:
    # 1) Initialize the River with your Nation as the Root
    my_river = RiverBlockTree("MyFirstNation")

    # 2) Build the 4 Pillars branches
    housing = my_river.add_pillar("Housing")
    banking = my_river.add_pillar("Banking")
    _food = my_river.add_pillar("Food Security")
    _education = my_river.add_pillar("Education")

    # 3) Run a "Flow" simulation (prime last-digit rhythm)
    prime_sequence = [1, 3, 7, 9, 1, 3]  # The 'rhythm' of your spiral

    print("--- Starting Sovereign River Flow ---")
    for digit in prime_sequence:
        apply_prime_gate(housing, digit)

        # Logic: If Housing is strong, it helps Banking
        if housing.resonance > 0.5:
            print(">>> Resonance Entanglement: Housing is boosting Banking...")
            apply_prime_gate(banking, 1)  # Standard boost


if __name__ == "__main__":
    main()
