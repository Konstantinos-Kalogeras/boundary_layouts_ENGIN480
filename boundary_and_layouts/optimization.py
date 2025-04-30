import numpy as np

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmProblem
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
import topfarm

from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014, Zong_PorteAgel_2020, Niayifar_PorteAgel_2016, CarbajoFuertes_etal_2018, Blondel_Cathelain_2020
from py_wake.utils.gradients import autograd
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site.shear import PowerShear
import pickle


with open(r'boundary_layouts_ENGIN480\boundary_and_layouts\utm_boundary.pkl', 'rb') as f:
    boundary = np.array(pickle.load(f))

with open(r'boundary_layouts_ENGIN480\boundary_and_layouts\utm_layout.pkl', 'rb') as f:
    xinit,yinit = np.array(pickle.load(f))


maxiter = 1000
tol = 1e-6

class V_1123(GenericWindTurbine):
    def __init__(self):
        """
        paramiters
        __________
        The turbulance intesity varies around 6-8%
        """
        # GenericWindTurbine.__init__(self, name = 'V_1123', diameter = 112,hub_height = 100,
                                    #    Power_norm = 3000, turbulance_intesity = 0.07)
        GenericWindTurbine.__init__(self, name='V_11-23', diameter=112, hub_height=100, 
                                    power_norm=3000, turbulence_intensity=0.07)


class EnedoLuchterdunenData(UniformWeibullSite):
    def __init__(self, ti= 0.07, shear=PowerShear(h_ref=100, alpha = 0.1)):
        f = [ 5.8007, 6.1557, 6.2208, 6.4858, 5.471, 5.4741, 
             7.7938, 13.2815, 16.8045, 10.4752, 8.6837, 7.3532]
        a = [7.95 ,    9.00 ,   9.45  ,  10.41 ,    8.87 ,    9.33   ,
              11.30 ,   12.62  ,  12.07   , 11.04   ,  9.49  ,   9.34]
        k = [2.002  ,  2.436 ,   2.662   , 2.533,    2.244,    2.291  ,
               2.205 ,   2.432 ,  2.260    ,2.127 ,   2.174,    2.068]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        # self.initial_position = np.array([site.x, site.y]).T
        self.name = 'Reovolution South Fork Wind'

wind_turbines = V_1123()

site = EnedoLuchterdunenData()

sim_res = Bastankhah_PorteAgel_2014(site, wind_turbines, k=0.0324555)

def aep_func(x,y):
    aep = sim_res(x,y).aep().sum()
    return aep
def daep_func(x,y):
    daep = sim_res.aep_gradients(gradient_method=autograd, wrt_arg=['x','y'], x=x,
                                y=y)
    return daep


boundary_closed = np.vstack([boundary, boundary[0]])


cost_comp = CostModelComponent(input_keys=['x', 'y'],
                                          n_wt = len(xinit),
                                          cost_function = aep_func,
                                          objective=True,
                                          maximize=True,
                                          output_keys=[('AEP', 0)]
                                          )


problem = TopFarmProblem(design_vars= {'x': xinit, 'y': yinit},
                         constraints=[XYBoundaryConstraint(boundary),
                                      SpacingConstraint(334)],
                        cost_comp=cost_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter, tol=tol),
                        n_wt=len(xinit),
                        expected_cost=0.001,
                        plot_comp=XYPlotComp()
                        )


cost, state, recorder = problem.optimize()

recorder.save('optimization_EnecoLutherduinenSite')

print('done')

print('done')

