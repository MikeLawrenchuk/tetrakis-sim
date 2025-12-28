def test_plot_floor_runs_without_error():
    from tetrakis_sim.defects import apply_blackhole_defect, find_event_horizon
    from tetrakis_sim.lattice import build_sheet
    from tetrakis_sim.plot import plot_floor_with_circle

    G = build_sheet(size=5, dim=3, layers=2)
    removed = apply_blackhole_defect(G, (2, 2, 1), 1.5)
    horizon = find_event_horizon(G, removed, 1.5, (2, 2, 1))
    plot_floor_with_circle(
        G,
        1,
        (2, 2, 1),
        1.5,
        highlight_nodes=removed,
        boundary_nodes=horizon,
        show=False,
    )
