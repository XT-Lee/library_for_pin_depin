import dissertation_plot as dp

ycm = dp.yukawa_crystal_module()
#ycm.set_crystal_type("bcc")
ycm.get_cystal_type()
ycm.lambda_water=0.01#um
for i in range(5):
    ycm.get_kappa_from_volume_fraction(i/5.0+1.0)