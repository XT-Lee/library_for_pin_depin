import points_analysis_2D_template as pat
import data_generator as dg
#txy = dg.gen_a_line()#checked right
txy = dg.gen_a_trajectory_with_oscillation_and_jump()
dtf = pat.data_transformation_from_trajectory_to_pca(txy)
cm = dtf.get_covariance_matrix_of_tx_from_averaged_position()
cmo = dtf.get_covariance_matrix_of_tx_from_initial_position()
print("jump")
print(f"cm: {cm}\n")
print(f"cmo: {cmo}\n")
txy = dg.gen_a_trajectory_with_period()
dtf = pat.data_transformation_from_trajectory_to_pca(txy)
cm = dtf.get_covariance_matrix_of_tx_from_averaged_position()
cmo = dtf.get_covariance_matrix_of_tx_from_initial_position()
print("period")
print(f"cm: {cm}\n")
print(f"cmo: {cmo}\n")