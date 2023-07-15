import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import optimize
import pickle
from time import gmtime, strftime
import generic_potential_modified
#from cosmoTransitions import tunneling1D
import tunneling1D_modified as tunneling1D
#from cosmoTransitions import pathDeformation as pd

############################################################################################

# Potential parameters that we want to scan over
#bvals = np.linspace(-4, -0.01, num=20)
#cvals = np.linspace(0, 16, num=20)
bvals = np.linspace(-2.0001, -2, num=2)
cvals = np.linspace(9.9999, 10, num=2)
Lamvals = np.array([1.])
c1vals = np.array([7.])
fvals = np.linspace(-10,10,21)

# Fixed parameters
gstar = 100

# params fed into potential
params = {
    'a': 0,
    'b': bvals,
    'c': cvals,
    'Lambda': Lamvals,
    'c1': c1vals,
    'f':fvals
}

#controlls if any and how many b and c values should be retried around failed points
recalc = True
recalc_all = False
recalc_resolution = 4
############################################################################################

class model1(generic_potential_modified.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.

    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self, a, b, c, Lambda, c1, f, **kwargs):
        """
          m1 - tree-level mass of first singlet when mu = 0.
          m2 - tree-level mass of second singlet when mu = 0.
          mu - mass coefficient for the mixing term.
          Y1 - Coupling of the extra boson to the two scalars individually
          Y2 - Coupling to the two scalars together: m^2 = Y2*s1*s2
          n - degrees of freedom of the boson that is coupling.
        """
        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 1
        self.deriv_order = 2
        self.Tmax = 2.

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
        self.a = a
        self.b = b
        self.c = c
        self.Lambda = Lambda
        self.c1 = c1
        self.f = f
        
    def approxZeroTMin(self):
        """
        Returns approximate values of the zero-temperature minima.
        This should be overridden by subclasses, although it is not strictly
        necessary if there is only one minimum at tree level. The precise values
        of the minima will later be found using :func:`scipy.optimize.fmin`.
        Returns
        -------
        minima : list
            A list of points of the approximate minima.
        """
        # This should be overridden.
        #CHRISTY NOTE: The origional code uses the default self.renormScaleSq = 1000.**2 as the approxZeroTMin. This is the default scale used in the MS-bar renormalization. Our minima are at much smaller scales, and so this function is being overwritten
        #return [np.ones(self.Ndim)*self.renormScaleSq**.5]
        
        approxZeroGuess = 1.**2
        return [np.ones(self.Ndim)*approxZeroGuess**.5]

    def Vtot(self, X, T, include_radiation=True):
        """
        This method defines the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        phi = X[...,0]
        
        # JACK NOTE: Here I wanted to avoid checking parameter space that we knew was not going to produce a 1st order PT
#         assert C < 0.5 + 9*B**2/(8*(1+4*A)), "no symmetry-breaking minimum"
#         assert C/(B+1e-12)**2 > 1, f"no phase transition; A {A}, B {B}, C {C}"
        width = 1
        w = (1-1/(1+np.exp(-self.c1*(phi-width/2))))/(1+np.exp(-self.c1*(phi+width/2)))
        w_0 = (1-1/(1+np.exp(-self.c1*(0-width/2))))/(1+np.exp(-self.c1*(0+width/2)))
        #w=0
        return ((-0.5 + self.c * T**2)*phi**2 + self.b * T*phi**3 + (0.25 + self.a)*phi**4 + self.f * T**4 * (w-w_0))

# scale-free potential without a window function:
def get_f_max_and_width(b, c):
    T_c = 1/np.sqrt(2*(c - b**2))
    phi_minus = (-3*b*T_c - np.sqrt(9*b**2*T_c**2 + 4*(1 - 2*c*T_c**2)))/2
    Vtot = (-.5 + c*T_c**2)*phi_minus**2 + b*T_c*phi_minus**3 + .25*phi_minus**4
    f_max = (Vtot/T_c**4)/10
    width = 2*phi_minus
    return f_max, width

for b in bvals:
    for c in cvals:
        print(get_f_max_and_width(b, c))
    
'''
def get_profile(phi_vals, V_vals, guesses=(5, 0)):
    f = interpolate.UnivariateSpline(phi_vals, V_vals, s=0, k=4)
    df = f.derivative(1)
    d2f = f.derivative(2)
    inst = tunneling1D.SingleFieldInstanton(*guesses, f, df, d2f)
    profile = inst.findProfile()
    action = inst.findAction(profile)
    
    return profile, action

def get_thermal_params(input_params):
    # Thermal parameters that we will derive (matrix with rows (first index) corresponding to b 
    # and columns (second index) corresponding to c
    # these values will be zero if CT chokes or the parameter space is invalid
    Tnucs = np.zeros((len(input_params['b']), len(input_params['c'])))
    betaHs = np.zeros((len(input_params['b']), len(input_params['c'])))
    ksis = np.zeros((len(input_params['b']), len(input_params['c'])))

    # we want to capture the output since CT really spits out a lot of info

    param_instance = input_params.copy()
    # loop over b vals
    for i, b in enumerate(input_params['b']):
        # idk why I did this but looks like just shifting to the center of the b bin for some reason
        param_instance['b'] = b
        # loop over c vals
        # note that our thermal parameter matrices will now be filled in, e.g., like ksis[i, j] = ...
        for j, c in enumerate(input_params['c']):
            # ditto
            param_instance['c'] = c

            # test the parameters, could put any restrictions we have on the potential here
            if c/(b**2) < 1:
                print('bad b,c', c, b, c/b**2)
                continue

            # print progress and turn off verbosity to spew out fewer details from CT
            print(i, j)
            try:
                m = model1(**param_instance)
                m.findAllTransitions(tunnelFromPhase_args={'verbose': False, 'fullTunneling_params': {'verbose': True}}); 
                #m.prettyPrintTnTrans()
            except:
                print('error')

            # Store the nucleation temperature
            try:
                if m.TnTrans == None:
                    print('no Tnuc calculated from CT')
                    continue
                elif np.size(m.TnTrans) == 0:
                    print('no Tnuc calculated from CT')
                    continue
                elif np.size(m.TnTrans) >= 1:
                    trans_i = -1
                    for Ti in range(np.size(m.TnTrans)):
                        if m.TnTrans[Ti]['low_vev'][0] - m.TnTrans[Ti]['high_vev'][0] > 0:
                            trans_i = Ti
                    if trans_i == -1:
                        print('no Tnuc because low_vev <= high_vev')
                        continue
                print('trans index:', trans_i)
                Tnuc = m.TnTrans[trans_i]['Tnuc']
                Tnucs[i, j] = Tnuc
            # if it didn't work because there was no PT, just move on to the next values
            # Note: this is probably a place to be careful, I'm assuming that there actually 
            # isn't a PT, but there could be and CT is just having trouble with the numerics
            except IndexError:
                print('no Tnuc', m.TnTrans)
                continue

            # maybe it found a 2nd order PT, in this case, we will ignore it!
            if m.TnTrans[trans_i]['trantype'] != 1:
                print('bad transition type', m.TnTrans[trans_i]['trantype'])
                continue

            # so now we have a FOPT with nucleation temp Tnuc, now we want to get a bit more info
            # we need to calculate the instanton for phi at various temperatures and take some derivatives using finite differences

            high_vev = m.TnTrans[trans_i]['high_vev'][0]
            low_vev = m.TnTrans[trans_i]['low_vev'][0]
            # we will calculate the potential at these field values (increase num to increase precision
            #phis = np.linspace(-5, m.TnTrans[trans_i]['low_vev']*2, num=20)
            phis = np.linspace(high_vev - np.abs(high_vev), low_vev + np.abs(low_vev), num=1000)
            # we will calculate the instanton at each of these temperatures (around Tnuc)
            Ts = np.linspace(Tnuc*0.999, Tnuc*1.001, num=100)
            # these are our initial guesses for the field minima (which should not be substantially far from those at Tnuc)
            # the fudge factors 1.01 and 0.99 are to ensure that tunneling1D can find these minima (i.e. they bound the region
            # that tunneling1D will look for minima), without fudge factors numerical hijinks can occur
            guesses = (low_vev + 0.01*np.abs(low_vev), high_vev - 0.01*np.abs(high_vev))

            # we will store all the details here
            profs, actions, good_Ts = [], [], []

            # loop through the temperatures
            for T in Ts:
                # get the potential at phis, T
                V_vals = m.Vtot(np.array([phis]).T, T)
                try:
                    # try to get the instanton profile and action
                    profile, action = get_profile(phis, V_vals, guesses=guesses)
                    profs.append(profile)
                    actions.append(action)
                    good_Ts.append(T)
                # if it fails...something is a little suss because we are very close to Tnuc, but sometimes things happen I guess
                except:
                    print(f"Temperature {T} has no symmetry-breaking global minimum")
                    pass

            # calculate thermal parameters
            # calculate beta
            try:
                ST = interpolate.UnivariateSpline(good_Ts, np.array(actions)/np.array(good_Ts), s=0, k=4)
                dSTdT = ST.derivative(1)
                #betaH = dSTdT(Tnuc)/Tnuc
                betaH = dSTdT(Tnuc)*Tnuc
                betaHs[i, j] = betaH
            except:
                print('no betas')

            # calculate ksi
            # need to compute deltaV
            deltaVs = []

            # this is a little bit awkward, but we loop through each instanton calculated at different Ts
            # and we calculate deltaV between the two minima
            # the reshaping is just an awkward parsing of CT's output
            for T, prof in zip(good_Ts, profs):
                deltaV = m.Vtot(np.reshape(prof.Phi[-1], (1, 1)), T) - m.Vtot(np.reshape(prof.Phi[0], (1, 1)), T)
                deltaVs.append(deltaV)

            # compute the delV as a function of T
            try:
                delV = interpolate.UnivariateSpline(good_Ts, deltaVs, s=0, k=4)
                ddelVdT = delV.derivative(1)

                #rho_n = np.pi**2 / 30 * gstar * Tnuc**4
                rho_n = np.pi**2 * 10 * Tnuc**4 / 3
                ksi = 1 / rho_n * (delV(Tnuc) - Tnuc * ddelVdT(Tnuc))

                ksis[i, j] = ksi
            except:
                print('no ksi')

            print('done with', i * len(input_params['b']) + c, '/', len(input_params['b']) * len(input_params['c']))
    
    return Tnucs, betaHs, ksis

#calculate thermal params
input_params = params.copy()
thermal_params = []
for Lambda in Lamvals:
    c1list = []
    for c1 in c1vals:
        flist = []
        for f in fvals:
            input_params['Lamda'] = Lambda
            input_params['c1'] = c1
            input_params['f'] = f
            Tnucs, betaHs, ksis = get_thermal_params(input_params)
            flist.append({'Tnucs': Tnucs, 'betaHs': betaHs, 'ksis': ksis})
        c1list.append(flist)
    thermal_params.append(c1list)
thermal_params = np.array(thermal_params)


#recalculate thermal params at higher resolution for points where first calculation failed
if recalc:
    for Li, Lambda in enumerate(Lamvals):
        for c1i, c1 in enumerate(c1vals):
            for fi, f in enumerate(fvals):
                for bi, b in enumerate(bvals):
                    for ci, c in enumerate(cvals):
                        if np.any((thermal_params[Li,c1i,fi]['Tnucs'][bi,ci]==0, 
                                   thermal_params[Li,c1i,fi]['betaHs'][bi,ci]==0, 
                                   thermal_params[Li,c1i,fi]['ksis'][bi,ci]==0)) and c/(b**2) >= 1 or recalc_all:
                            input_params['Lamda'] = Lambda
                            input_params['c1'] = c1
                            input_params['f'] = f
                            b_step = bvals[1]-bvals[0]
                            c_step = cvals[1]-cvals[0]
                            input_params['b'] = np.linspace(b - b_step/2, b + b_step/2, recalc_resolution)
                            input_params['c'] = np.linspace(c - c_step/2, c + c_step/2, recalc_resolution)
                            Tnucs, betaHs, ksis = get_thermal_params(input_params)
                            Tnucs_mean = Tnucs[Tnucs != 0].mean()
                            betaHs_mean = betaHs[betaHs != 0].mean()
                            ksis_mean = ksis[ksis != 0].mean()
                            if Tnucs_mean == np.nan:
                                Tnucs_mean = 0
                            if betaHs_mean == np.nan:
                                betaHs_mean = 0
                            if ksis_mean == np.nan:
                                ksis_mean = 0
                            thermal_params[Li,c1i,fi]['Tnucs'][bi,ci] = Tnucs_mean
                            thermal_params[Li,c1i,fi]['betaHs'][bi,ci] = betaHs_mean
                            thermal_params[Li,c1i,fi]['ksis'][bi,ci] = ksis_mean

    
# output file
print('#saving output file#')
results = (params, thermal_params)
print(thermal_params)
pickle.dump(results, open('./cosmotrans_out/' + strftime("%Y_%m_%d %H_%M_%S", gmtime()) + '.pk', 'wb'))
'''