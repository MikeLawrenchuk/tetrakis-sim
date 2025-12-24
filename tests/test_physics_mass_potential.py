import networkx as nx

from tetrakis_sim.physics import run_wave_sim


def test_mass_scales_laplacian_response():
    """
    With a larger mass on a node, the Laplacian forcing term should be reduced
    because the update uses (coeff / mass) * lap.
    """
    dt = 0.1

    # Baseline (mass defaults to 1.0)
    G1 = nx.Graph()
    G1.add_edge(0, 1)
    h1 = run_wave_sim(G1, steps=1, initial_data={0: 1.0}, c=1.0, dt=dt, damping=0.0)
    u0_mass1 = h1[-1][0]

    # Larger mass at node 0
    G2 = nx.Graph()
    G2.add_edge(0, 1)
    G2.nodes[0]["mass"] = 10.0
    h2 = run_wave_sim(G2, steps=1, initial_data={0: 1.0}, c=1.0, dt=dt, damping=0.0)
    u0_mass10 = h2[-1][0]

    # With high mass, node 0 should change *less* (stay closer to 1.0),
    # hence be larger than the mass=1 baseline after one step.
    assert u0_mass10 > u0_mass1


def test_potential_adds_local_restoring_term():
    """
    With a positive potential on a node, the update includes -coeff * potential * u[n],
    so the amplitude at that node should be reduced relative to potential=0.
    """
    dt = 0.1

    # Baseline (potential defaults to 0.0)
    G0 = nx.Graph()
    G0.add_edge(0, 1)
    h0 = run_wave_sim(G0, steps=1, initial_data={0: 1.0}, c=1.0, dt=dt, damping=0.0)
    u0_pot0 = h0[-1][0]

    # Positive potential at node 0
    Gp = nx.Graph()
    Gp.add_edge(0, 1)
    Gp.nodes[0]["potential"] = 2.0
    hp = run_wave_sim(Gp, steps=1, initial_data={0: 1.0}, c=1.0, dt=dt, damping=0.0)
    u0_pot2 = hp[-1][0]

    assert u0_pot2 < u0_pot0
