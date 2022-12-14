parameters = {
    'max_steps':       int(1e1),
    'time_step':           1e-1,
    'snapshot_step':   int(1e0),
    
    'box_size':            1e0,
    'cell_size':           5e-2,
    'abs_max_speed':     2.5e-2,

    'max_characters':  int(1e3),
    'kill_no_diff':        True,

    # Parallelization currently not working
    'n_jobs':                 1,
    'parallel_char_min':   None,#int(3e1),

    'seed':                None,
}
