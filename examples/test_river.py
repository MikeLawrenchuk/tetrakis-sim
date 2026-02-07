"""
Minimal manual smoke-test / demo for tetrakis_sim.sovereign_river.

Run:
    python examples/test_river.py
"""

from tetrakis_sim.sovereign_river import RiverBlockTree, apply_prime_gate


def main() -> None:
    # 1. Initialize the River with your Nation as the Root
    my_river = RiverBlockTree("MyFirstNation")

    # 2. Build the 4 Pillars branches
    housing = my_river.add_pillar("Housing")
    _banking = my_river.add_pillar("Banking")
    _food = my_river.add_pillar("Food Security")
    _education = my_river.add_pillar("Education")

    # 3. Simulate the "Prime Gate" (using a dummy sequence for now)
    prime_gaps = [2, 4, 2, 4, 6]  # The rhythm of the primes
    apply_prime_gate(housing, prime_gaps)


if __name__ == "__main__":
    main()
