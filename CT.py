import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pickle
from time import gmtime, strftime
from cosmoTransitions import generic_potential
from cosmoTransitions import tunneling1D
#from cosmoTransitions import pathDeformation as pd

############################################################################################

# Potential parameters that we want to scan over
bvals = np.linspace(-4, 0, num=2)
cvals = np.linspace(0, 16, num=2)

# Fixed parameters
Lamvals = 1. 
gstar = 100

# params fed into potential
params = {
    'Lambda': Lamvals,
    'a': 0,
    'b': bvals,
    'c': cvals
}
############################################################################################

class model1(generic_potential.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.

    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self, a, b, c, Lambda, **kwargs):
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
        self.Tmax = 10

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
        self.a = a
        self.b = b
        self.c = c
        self.Lambda = Lambda

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
        f = 1
        c1 = 7
        w = (1-1/(1+np.exp(-c1*(phi-.5))))/(1+np.exp(-c1*(phi+.5)))
        #w=0
        return ((-0.5 + self.c * T**2)*phi**2 + self.b * T*phi**3 + (0.25 + self.a)*phi**4 + f * T**4 * w)

def get_profile(phi_vals, V_vals, guesses=(5, 0)):
    f = interpolate.UnivariateSpline(phi_vals, V_vals, s=0, k=4)
#     print(f(5))
    df = f.derivative(1)
    d2f = f.derivative(2)
    inst = tunneling1D.SingleFieldInstanton(*guesses, f, df, d2f)
    profile = inst.findProfile()
    action = inst.findAction(profile)
    
    return profile, action


# Thermal parameters that we will derive (matrix with rows (first index) corresponding to b 
# and columns (second index) corresponding to c
# these values will be zero if CT chokes or the parameter space is invalid
Tnucs = np.zeros((len(bvals), len(cvals)))
betaHs = np.zeros((len(bvals), len(cvals)))
ksis = np.zeros((len(bvals), len(cvals)))

# we want to capture the output since CT really spits out a lot of info

# loop over b vals
for i, b in enumerate(bvals):
    # idr why I did this but looks like just shifting to the center of the b bin for some reason
    params['b'] = b + bvals[1] - bvals[0]
    # loop over c vals
    # note that our thermal parameter matrices will now be filled in, e.g., like ksis[i, j] = ...
    for j, c in enumerate(cvals):
        # ditto
        params['c'] = c + cvals[1] - cvals[0]

        # test the parameters, could put any restrictions we have on the potential here
        if c/(b**2) < 1:
            print('bad b,c', c, b, c/b**2)
            continue

        # print progress and turn off verbosity to spew out fewer details from CT
        print(i, j)
        try:
            m = model1(**params)
            m.findAllTransitions(tunnelFromPhase_args={'verbose': False, 'fullTunneling_params': {'verbose': False}}); 
            m.prettyPrintTnTrans()
        except:
            print('error')

        # Store the nucleation temperature
        try:
            if np.size(m.TnTrans) == 2:
                trans_i = 1
            elif np.size(m.TnTrans) == 1:
                trans_i = 0
                print('no Tnuc')
                continue
            elif np.size(m.TnTrans) == 0:
                print('no Tnuc')
                continue
            if m.TnTrans == None:
                print('no Tnuc')
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

        # we will calculate the potential at these field values (increase num to increase precision
        phis = np.linspace(-5, m.TnTrans[trans_i]['low_vev']*2, num=20)
        # we will calculate the instanton at each of these temperatures (around Tnuc)
        Ts = np.linspace(Tnuc*0.999, Tnuc*1.001, num=20)
        # these are our initial guesses for the field minima (which should not be substantially far from those at Tnuc)
        # the fudge factors 1.01 and 0.99 are to ensure that tunneling1D can find these minima (i.e. they bound the region
        # that tunneling1D will look for minima), without fudge factors numerical hijinks can occur
        guesses = (m.TnTrans[trans_i]['low_vev'][0]*1.01, m.TnTrans[trans_i]['high_vev'][0]*0.99)

        # we will store all the details here
        profs, actions, good_Ts = [], [], []

        # loop through the temperatures
        for T in Ts:
            # get the potential at phis, T
            V_vals = m.Vtot(phis, T)
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
            betaH = dSTdT(Tnuc)/Tnuc
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

            rho_n = np.pi**2 / 30 * gstar * Tnuc**4
            ksi = 1 / rho_n * (delV(Tnuc) - Tnuc * ddelVdT(Tnuc))

            ksis[i, j] = ksi
        except:
            print('no ksi')

        print('done with', i * len(bvals) + c, '/', len(bvals) * len(cvals))


# output file
thermal_params = {
    'Tnucs': Tnucs,
    'betaHs': betaHs,
    'ksis': ksis}

results = (params, thermal_params)

# write output to files should you so desires
pickle.dump(results, open('./cosmotrans_out/' + strftime("%Y_%m_%d %H_%M_%S", gmtime()), 'wb'))

