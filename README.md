# library_for_pin_depin
## description:
- this is just a library for me to record the codes used to finish my Ph.D project.
- the repository has a series of workflows to operate HOOMD-blue; 
- and has a library to proceed trajectories of particles in 2D and get their structural or kinetic properties
## file structure:
- this library contains four parts: HOOMD-blue simulation module, data management module, data analysis module and application scripts. 
- the filenames are listed as follows.
- HOOMD-blue simulation module:
	all files in symmetry_transformation.
- data management module:
	opertateOnMysql,
	proceed_file.
- data analysis module:
	points_analysis_2D,
	particle_tracking.
	particle_decorate
	data_decorate
- application scripts:
	list_code_analysis
	analysis_data_from_image
	workflow_part,
	workflow_analysis,
	data_analysis_cycle,
	getDataAndDiagram,
	and others.
## get exp_data
- data_analysis_cycle.save_exp_20230113_6 #video_bond_traps
- workflow_analysis.show_bond_image
- particle_tracking.
- particle_decorate

## log:
- 20240115 edit workflow_analysis.show_bond_image.read_image,draw_bond_exp_hex
			add points_analysis_2D.bond_plot_module.plot_scale_bar
- 20240111 add operate_simulation_langevin_wca_yukawa in simulation_core_p.simulation_core_traps
			add simulation_controller_wca_yukawa in simulation_controller_p
			add set_new_gsd_file in system_parameters_generators.initial_state_generator
- 20240105 add example_plot.generate_hertzian,generate_kob_andersen
- 20231227 add generate_simu_index_csv_depin_special_low_t in 
			symmetry_transformation_v4_3.simulation_controller.simulation_controller_type_n_part_traps
- 20231226 add analysis_data_from_image, add_traps_to_points_kagome_part, draw_points_kagome_part,
			 add_traps_to_points_honeycomb_part, draw_points_honeycomb_part
- 20231222 add get_cnks_from_csv_type_4569_part, get_cnks_from_csv_files_type_n_part in 
			symmetry_transformation_v4_3.list_code_analysis.analyze_a_series_of_gsd_file
- 20231221 fix the problem of adding an unnamed column in symmetry_transformation_v4_3.list_code_analysis.
        	analyze_a_series_of_gsd_fileget_cnks_from_csv_files_type_n
- 20231219 edit generate_initial_state_hex_particle_XXX_trap_scan_csv, col[i] -> 'col_name' to avoid new unnamed column:0.
- 20231217 add generate_initial_gsd_hex_and_type_n in symmetry_transformation_v4_3.simulation_controller
- 20231215 add get_cn5s_from_csv_files in symmetry_transformation_v4_3.list_code_analysis.analyze_a_series_of_gsd_file
			edit get_cn_k_from_a_gsd_frame in symmetry_transformation_v4_3.analysis_controller.get_data_from_a_gsd_frame
- 20231214 add generate_simu_index_csv_type_10_pin_3_30 in 
			symmetry_transformation_v4_3.simulation_controller.simulation_controller_type_n_part_traps
- 20231213 add set_new_gsd_file_2types_by_box_or_n_size
			in symmetry_transformation_v4_3.system_parameters_generators.initial_state_generator
			edit generate_gaussian in function_plot.example_plot.functions_plot_module
- 20231212 add generate_initial_state_hexagonal_particle_honeycomb_part_gaus_eq_harmo_trap_scan_csv in 
			symmetry_transformation_v4_3.simulation_controller
			add generate_gaussian_force, generate_harmonic_force, generate_gaussian, generate_harmonic in example_plot
			add get_bonds_from_simu_indices_type_n_from_csv in symmetry_transformation_v4_3.list_code_analysis.analyze_a_series_of_gsd_file
- 20231211 add get_2_k_cn3_stds_and_plot,get_cn3s_from_two_csv_files 
			in symmetry_transformation_v4_3.list_code_analysis.analyze_a_series_of_gsd_file
			add merge_two_csvs in proceed_file.
			edit set_simulation_parameters self.width = 1 in simulation_core.simulation_core_traps
- 20231208 clear bond_plot_module_old in points_analysis_2D.
			add workflow_type_n in workflow_analysis.archimedean_tilings_polygon_dye.
			edit perturbation, fix positions.z = 0  in system_parameters_generators.initial_state_generator.set_new_gsd_file_2types_by_n_size
- 20231207 add generate_type_n_part,generate_type_n in workflow_analysis.archimedean_tilings
- 20231204 add trap_fill_box in system_parameters_generators.initial_state_generator.set_new_gsd_file_2types
			add get_gsds_from_mysql.get_record_from_csv
- 20231130 add get_conditional_bonds_and_simplices_bond_length in points_analysis_2D.static_points_analysis_2d.
			add workflow_analysis.archimedean_tilings_polygon_dye
			edit workflow_analysis.archimedean_tilings
- 20231127 add  loadCsvDataToMysql, loadTxtDataToMysql in opertateOnMysql
- 20231124 add archimedean_tilings_polygon_dye in workflow_analysis.
			add generate_simu_index_csv in symmetry_transformation_v4_3.simulation_controller
			add generate_initial_state_hexagonal_particle_honeycomb_trap_scan_csv in symmetry_transformation_v4_3.simulation_controller
- 20231123 add cut_edge_of_positions_manual in points_analysis_2D.static_points_analysis_2d
- 20231122 edit simulation_controller.simulation_controller_traps.generate_initial_state_hexagonal_particle_honeycomb_trap
			add simulation_controller.simulation_controller_traps.generate_initial_state_hexagonal_particle_honeycomb_trap_scan_k
- 20231120 edit get_trajectory_data in proceed_file.proceed_gsd_file
			edit test_simulation
- 20231117 add generate_type10_part,generate_type10 in  workflow_analysis.archimedean_tilings.
			add set_new_gsd_file_2types in system_parameters_generators.initial_state_generator.
- 20231116 add archimedean_tilings in  workflow_analysis.
			add generate_type11_part,generate_type11 in  workflow_analysis.archimedean_tilings.
- 20231115 add show_bonds_transition_from_hex_to_kagome in workflow_analysis.
			edit draw_points_with_given_bonds in points_analysis_2D.bond_plot_module for matplotlib update.
- 20231113 add symmetry_transformation_v4_3 to adjust the new hoomd.
- 20231107 add get_cn3_vs_u_sub_low_depin,txt_to_data_low_depin in workflow_analysis.controller_get_honey_cn3_vs_u_sub.
			add in workflow_analysis.controller_get_honey_part_cn3_vs_u_sub
			add saveIndexkTCN4CN3depin in data_analysis_cycle
- 20231106 add saveIndexkTCN4CN3SeedsPsi6 in data_analysis_cycle
- 20231027 add extend_traps_to_9boxes in workflow_analysis.workflow_temp
- 20231026 edit cut_edge_of_positions_by_box in points_analysis_2D.static_points_analysis_2d
- 20231025 add cut_edge_of_positions_by_box in points_analysis_2D.static_points_analysis_2d
- 20231019 add search_and_get_trajectory, search_and_get_single_final_frame in workflow_analysis.workflow_file_to_result.
			get_bond_trap_plot_param_compare
- 20231017 edit compute_energy_particle_wise in points_analysis_2D.energy_computer. 
- 20231016 decouple get_extended_positions_from_points and get_extended_positions in proceed_file.proceed_gsd_file
			add compute_energy_particle_wise. in .
- 20231014 add interact_mode in points_analysis_2D.energy_computer.compute_distance
- 20231013 edit read_a_frame, adding dimension, in proceed_file.proceed_gsd_file.
			add get_extended_positions in proceed_file.proceed_gsd_file
- 20231012 edit the introduction of workflow_analysis.
			add data_retriever.
			add energy_computer to calculate the free energy of a system in in points_analysis_2d.
- 20231010 add array_to_xyz in proceed_file.data_type_transformer.
- 20230907 add plot_to_check_trans_ratio,plot_to_check_act_ratio,plot_to_check_act_share_ratio in workflow_analysis.show_pin_interstitial_order_parameter.
			add get_pin_depin_event in points_analysis_2d.dynamical_facilitation_module
- 20230906 add show_pin_interstitial_order_parameter in workflow_analysis
			add data_decorate
			add data_decorator in data_decorate
			add coarse_grainize_data_log,coarse_grainize_and_average_data_log in data_decorate.data_decorator
- 20230905 add get_interstitial_bool in points_analysis_2d.dynamical_facilitation_module
- 20230826 edit example in workflow_part.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m
			add particle_decorate.py
- 20230814 edit workflow_part.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m
			edit control_table
- 20230807 edit get_cost_function_cluster_ml
- 20230806 edit get_cost_function_cluster_ml
- 20230804 add polygon_analyzer_ml to reorganize get_conditional_bonds_and_simplices_ml and get_cost_function_cluster_ml
			add __init__() to relink vertices of rational ridge length in points_analysis_2d.static_points_analysis_2d.polygon_analyzer_ml
			edit two modes in points_analysis_2d.static_points_analysis_2d.polygon_analyzer_ml.get_conditional_bonds_and_simplices_ml
- 20230803 edit get_cost_function_cluster_ml
- 20230731 edit record_vertices_cluster[vertex_id, cluster_id], not initialize cluster_id as -1 but range(n_vertices)
			in points_analysis_2d.static_points_analysis_2d.get_conditional_bonds_and_simplices_ml.
			caution: edges not cut ! [x] in points_analysis_2d.static_points_analysis_2d.get_cost_function_cluster_ml.
			codes are finished but not checked [x]
- 20230730 edit get_conditional_bonds_and_simplices_ml in points_analysis_2d.static_points_analysis_2d[on going]
- 20230729 add account, filename_seed in data_analysis_cycle.saveIndexCN346PCairoSeed
			add account in workflow_part.workflow_simu_to_mysql_pin_hex_to_cairo_egct_uniform
			add get_conditional_bonds_and_simplices_ml in points_analysis_2d.static_points_analysis_2d
			add get_cost_function_cluster_ml in points_analysis_2d.static_points_analysis_2d
- 20230728 edit steps set option in pin_seed_oop.workflow_uniform.__init__().
			edit pcairo1 in data_analysis_cycle.saveIndexCN346PCairoSeed
			solve passing over month bug in computeTime.getTimeCost
- 20230727 add example for get_trajectory_data in proceed_file.proceed_gsd_file
			rename function __init_state_launch as __launch_init_state in pin_seed_oop.
- 20230726 edit str_index=str(int(simu_index)) in data_analysis_cycle.save_from_gsd 
			edit get_trajectory_data, add simu_index1 in proceed_file.proceed_gsd_file
			edit scp = pa.show_cairo_order_parameter(), overplot 'r+' in data_analysis_cycle.save_from_gsd(if p_cairo)
- 20230725 add introduction for compute_cairo_order_parameter in workflow_analysis.show_cairo_order_parameter.
			add get_cairo_order_parameter in points_analysis_2d.static_points_analysis_2d. link to show_cairo_order_parameter.
			move and edit show_cairo_order_parameter from workflow_analysis to  points_analysis_2d.
- 20230724 add plot_bond_ridge_rank_idea 
			in points_analysis_2d.static_points_analysis_2d, 
			which is an idea of bond plot, 
			bond_length_rank vs ridge_length_rank.
			add get_bonds_with_machine_learning_idea
			in points_analysis_2d.bond_plot_module [ongoing]
- 20230713 edit trajectory_coarse_grain_general in points_analysis_2d.trajectory_module
			add plot_bicolor_final_displacements_static in points_analysis_2d.trajectory_module
			add bond_plot_for_ai, particle large, trap small[ongoing]
			edit bond_plot_module.draw_points_with_given_bonds,plot_traps,
			draw_points_with_given_bonds in points_analysis_2d.
- 20230712 edit workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m precise  in workflow_part
scatt.workflow_mysql_to_data_pin_hex_to_kagome_klt_2m('remote')
- 20230711 edit plot_h_hp_cn3_scan_k_fast, add shallow honeycomb pin filename, in workflow_analysis.compare_diagram_dynamics
			add save_cn3_decorate_to_csv, save_cn3_decorate_to_csv_unit in workflow_analysis.compare_diagram_dynamics
			edit workflow_data_to_diagram_kagome_mark pin_check_s in  getDataAndDiagram
- 20230710 add data_type_transformer.array_to_csv in proceed_file.
			edit save_from_gsd_to_cn3 in data_analysis_cycle
			edit record_lk36shallow = in getDataAndDiagram
- 20230707 edit draw_polygon_patch_oop polygon_color in points_analysis_2d.static_points_analysis_2d.
			edit trajectory_coarse_grain_general color_pin,color_inte in points_analysis_2d.trajectory_module
- 20230706 edit points_analysis_2d.trajectory_module.		
			plot_bicolor_final_displacements, tune color.
			edit data_analysis_cycle.saveIndexklTCN3CN4Seed prefix_gsd -> prefix
			edit workflow_part.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m k1 stp kend
- 20230703 edit save_from_gsd.bond_plot.draw_bonds_conditional_bond as draw_bonds_conditional_bond_oop
			add show_final_frame.search_lcr_k in workflow_analysis[finish]
			edit workflow_part.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m
			edit data_analysis_cycle.saveIndexklTCN3CN4Seed to redirect prefix_gsd.
- 20230701 add show_final_frame.search_lcr_k in workflow_analysis[ongoing]
- 20230619 diagram sseries finished.
			edit workflow_part.workflow_simu_to_mysql_pin_hex_to_kagome
			edit symmetry_transformation.symmetry_transformation_auto_kagome_pin
			edit saveIndexCN4CN6SeedPsi6
- 20230615 add get_diagram_data_mark_multi in getDataAndDiagram.mysql_data_processor
			add workflow_data_to_diagram_honeycomb_mark,
			workflow_mysql_to_data_depin_from_honeycomb_mark,
			workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m_random_mark
			in getDataAndDiagram
			add workflow_data_to_diagram_kagome_mark,
			workflow_mysql_to_data_depin_from_kagome1_mark,
			workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_random_mark
			in getDataAndDiagram 
			add workflow_data_to_diagram_kagome_part_mark,
			workflow_mysql_to_data_depin_from_kagome_part1_mark,
			workflow_mysql_to_data_pin_hex_to_kagome_part_klt_2m_random_mark
			in getDataAndDiagram
- 20230614 add draw_diagram_scatter_mark_multi in getDataAndDiagram.mysql_data_processor
			add workflow_data_to_diagram_honeycomb_part_mark,
			workflow_mysql_to_data_depin_from_honeycomb_part1_mark,
			workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_mark
			in getDataAndDiagram
- 20230613 add workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_mark
			in getDataAndDiagram
- 20230612 redirect log_prefix to 2T disk in symmetry_transformation.pin_seed_oop.
- 20230610 add showTables in opertateOnMysql
- 20230609 add select_content in opertateOnMysql.getDataFromMysql
			add workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m_random in getDataAndDiagram
- 20230605 add dis_edge_cut in static_points_analysis_2d.__cut_edge_of_positions
			add dis_edge_cut in data_analysis_cycle.saveIndexCN346PCairoSeed
			add createTableInMysql in opertateOnMysql
- 20230603 add plot_diagram_value in getDataAndScatter
			rename getDataAndScatter as getDataAndDiagram
- 20230602 add data_decorating in workflow_analysis.compare_diagram_dynamics.
- 20230526 edit plot_polygon_bond_xylim in workflow_analysis.show_polygon_dye
			rename show_tuned_image as show_bond_image
- 20230525 edit trajectory_coarse_grain_single_particle in workflow_analysis.show_waiting_time_interstitial_motion
			add trajectory_module in points_analysis_2d
			edit get_points_plot_loop in workflow_analysis.show_polygon_dye
- 20230523 add self.image in particle_tracking.particle_track.single_frame_particle_tracking 
			add file_name parameters in workflow_analysis.draw_tuned_image
			add axis_limit in particle_tracking.particle_track.single_frame_particle_tracking
- 20230519 add "ridge_length_method" in 
			points_analysis_2d.static_points_analysis_2d.get_coordination_number_conditional
- 20230517 add first_two_peak_group_method in 
			points_analysis_2d.static_points_analysis_2d.get_first_minima_bond_length_distribution
			edit "global_compare_method" to fix bug in 4323_8_2000
- 20230517 add compare_diagram_dynamics.plot_h_hp_cn3_scan_k_fast
		    add compare_diagram_dynamics.get_data_cn3_scan_k_fast
- 20230516 simu is forced to stop 
- 20230515 add compare_diagram_dynamics.plot_cn3_scan_k_fast(rename as plot_hp_cn3_scan_k_fast)
		    add compare_diagram_dynamics.plot_honey_cn3_scan_k
- 20230512 edit workflow_simu_to_mysql_pin_hex_to_honeycomb_oop_klt_2m, check mode.[v]
- 20230511 add compare_diagram_dynamics.plot_cn3_scan_lcr
- 20230510 add compare_diagram.
- 20230508 add workflow_analysis.compare_diagram
	   edit test_cuda.comapre_speed_cpu_vs_gpu
- 20230504 add file_for_CUDA.test_cuda
- 20230424 edit workflow_analysis.show_disp_field.get_bicolor_disp.
- 20230423 add workflow_part.saveIndexCN346PCairoSeed. 
- 20230422 add workflow_analysis.show_cairo_order_parameter.
- 20230420 add proceed_file to operate files and make directories.
- 20230420 fix get_conditional_bonds_and_simplices, return count_polygon_relative
- 20230410 edit get_conditional_bonds_and_simplices, return count_polygon_relative
- 20230408 edit bond_plot_module init(ax=none)
- 20230404 add points_analysis_2d.count_polygon_n
- 20230329 add workflow_analysis.show_waiting_time_interstitial_motion
	   add points_analysis_2d.dynamical_facilitation_module.scan_displacement_t
- 20230328 add workflow_analysis.optimize_polyfit
	   add workflow_analysis.show_waiting_time_brownian
	   add workflow_analysis.show_waiting_time_dynamical_facilitation
- 20230324 add workflow_analysis.show_tuned_image to add line over image.
	   add bond_plot_module_for_image in points_analysis_2d
- 20230320 add workflow_uniform_continue in symmetry_transformation.simple_simulation
	   add workflow_analysis.get_msd_from_gsd()
	   add workflow_analysis.get_displacement_field_normal()
- 20230314 add workflow_analysis to record codes.
- 20230306 plot scatter vacancy and occupation in dynamical_facilitation_module.plot_reference_occupation
	   add dynamical_facilitation_module.plot_reference_positions_waiting_time_colored
- 20230225 set honeycomb3-8-12 as facilitation reference, part1 as traps.
- 20230223 add probability distribution of dynamical facilitation(ongoing)
	   add get_displacement_1D_overlap in data_analysis_cycle
	   add dynamical_facilitation_module in points_analysis_2d
- 20230222 add function_plot including example_plot and unit_transform
- 20230216 add workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_oop
	   add mysql_data_processor in getDataAndScatter
	   add get_string_like_motion_rank in data_analysis_workflow, 
	   add plot_string_like_motion_rank in dynamic_points_analysis_2d
	   add get_displacement_field_xy_rank in displacemnt_field_2D
- 20230206 seperate draw_bonds_conditional_bond() into two parts( listed as follows) in bond_plot_module in points_analysis_2D
	   part1, restrict_axis_property()
	   part2, draw_points_with_conditional_bond()
- 20230131 add workflow_liquid in workflow_part
	   add init_cut in dynamic_points_analysis_2d.plot_bond_neighbor_change_oop
	   add __init_state_pin_from_gsd etc in pin_seed_oop
	   add lattice_constant in coordination_number module in save_from_gsd 
- 20230129 edit example in workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m()
	   pip install latex, code added in bond_plot_module in points_analysis_2D
- 20230119 edit workflow_part.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m()
	   edit particle_tracking
- 20230118 edit data_analysis_cycle.data_analysis_workflow.get_displacment_field() 
	   generalize displacment_field_2D.get_displacement_field_xy() to any two frames.
	   add savetxt function in plot_hist_neighbor_change_event() in points_analysis_2D.dynamic_points_analysis_2d
	   add check mode in workflow_part.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m()
	   add zorder to rank plot object in figure, in 
- 20230116 edit data_analysis.txyz_to_bond_plot,bond_plot_module
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
