# library_for_pin_depin
## description:
- this is just a library for me to record the codes used to finish my Ph.D project.
- the repository has a series of workflows to operate hoomd-blue; 
- and has a library to proceed trajectories of particles in 2D and get their structural or dynamic properties.

## log:
- 20230116 edit data_analysis.txyz_to_bond_plot,
	   dynamic_points_analysis_2d.plot_bond_neighbor_change_oop,
	   static_points_analysis_2d.draw_bonds_conditional_bond_oop,
	   create class bond_plot_module in points_analysis_2D
- 20230107 simplify pin_seed_oop
	   edit data_analysis_cycle.data_analysis
	   edit workflow_part.workflow_simu_to_mysql_pin_hex_to_honeycomb_oop_klt_2m.
	   edit getDataAndScatter.
- 20230106 edit dynamic_points_analysis_2d.compute_nearest_neighbor_displacements & dynamic_coordination_bond_plot,
	   static_points_analysis_2d.get_first_minima_bond_length_distribution & draw_bonds_conditional_bond in points_analysis_2d_CUDA
	   fix error: 'ndarray' object cannot be interpreted as an integer
	   fix error: 'BlockManager' object has no attribute 'columns'
	   the main time cost is save numerous images.
- 20230105 fix bug of saving txyz as txyz_stable into 'txyz_stable.npy'
	   update get_displacement_field_xy.given txyz_stable, cut_edge is not necessary anymore.
	   move displacement_field_xy from proceed_gsd_file to dynamic_points_analysis_2d in points_analysis_2d
	   it seems that while-loop proceeding static_points_analysis_2d is too slow.
	   Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a NumPy array explicitly.
	   https://cloud.tencent.com/developer/ask/sof/927734
- 20230104 reorganize structure of points_analysis_2D.dynamic_points_analysis_2d
	   edit proceed_gsd_file, dynamic_points_analysis_2d in points_analysis_2D
- 20230103 add monitor_neighbor_change_event(), bond_plot(), hist_neighbor_change_event() in points_analysis_2D.dynamic_points_analysis_2d
- 20230102 add sum_id_neighbors in static_points_analysis_2d.get_nearest_neighbor_dispalcement(), to check if neighbors of a center particle have changed over time.
- 20230101 set new standard of getDataAndScatter
- 20221219 invert the pixel coordination from image y-axis to coordination y-axis in particle_tracking.folder_frames_particle_tracking
	   test data_analysis_cycle.save_points_from_exp, right.
	   test and correct points_analysis_2d.dynamic_points_analysis_2d.compute_msd_normal, compute_atmsd_scan_t, compute_atmsd_t_chips
- 20221216 add data_analysis_cycle.save_points_from_exp, waiting test.
- 20221215 add Lindemann msd in dynamic_points_analysis_2d.
	   rename points_analysis_2D as dynamic_points_analysis_2d
- 20221213 add points_analysis_2D.msd.compute_nn_msd()
- 20221212 add introduction, mode='simu'or'exp', plot_msd,plot_trajectory, for points_analysis_2D.msd
- 20221209 edit points_analysis_2D.msd 
- 20221208 add kagome/honeycomb pin precise/low T
- 20221203 add seed in points_analysis_2D.proceed_gsd_file
	   add points_analysis_2D.msd ongoing
- 20221202 cut edge in workflow_part.workflow_simu_to_mysql_depin_from_kagome;
	   cut edge in workflow_part.workflow_simu_to_mysql_depin_from_honeycomb_part,
	   add RandomSeed in data_analysis_cycle.saveIndexPsi3Psi6;
	   cut edge in melt_hex_from_honeycomb;
	   add simple_simulation;
	   edit account in points_analysis_2D;
- 20221130 add psik_plot in data_analysis_cycle.save_from_gsd; 
	   add psi6 in workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m;
	   add condition="where kT = 1" in workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m(account='tplab')
- 20221129 point_analysis_2D mean repalced by median; symmetry_transformation/pin_seed_oop.py add mode=""
- 20221128 pin_seed_oop.py add self.mode
- 20221128 add file get_a_from_image
