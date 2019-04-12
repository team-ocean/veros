"""
Functions calculating atmosphere-ocean fluxes

"""
from veros import veros_method
#import numpy as np  # TODO remove this as veros handles it
from . import cyclic


@veros_method
def carbon_flux(vs):
    t_in = vs.temp[:, :, -1, vs.tau]  # [degree C] TODO rename variable
    s_in = vs.salt[:, :, -1, vs.tau]  # [g/kg = PSU] TODO rename variable
    dic_in = vs.dic[:, :, -1, vs.tau] * 1e-3# [mmol -> mol] TODO rename variable
    ta_in = vs.alkalinity[:, :, -1, vs.tau] * 1e-3 # [mmol -> mol] TODO rename variable
    co2_in = vs.atmospheric_co2 # [ppmv] TODO rename variable

    # ao = 1  # 1 - ice fraction coverage
    icemask = np.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
            vs.forc_temp_surface < 0.0)
    ao = np.logical_not(icemask)


    atmospheric_pressure = 1  # atm  NOTE: We don't have an atmosphere yet, hence constant pressure

    # TODO get actual wind speed rather than deriving from wind stress
    wind_speed = np.sqrt(np.abs(vs.surface_taux) + np.abs(vs.surface_tauy)) * 500 # mult for scale?
    vs.wind_speed = wind_speed


# !     xconv is constant to convert piston_vel from cm/hr -> cm/s
# !     here it is 100.*a*xconv (100 => m to cm, a=0.337, xconv=1/3.6e+05)
#       xconv = 33.7/3.6e+05
#       xconv = xconv*0.75
    xconv = 0.337 / 3.6e5
    xconv *= 0.75  # NOTE: This seems like an approximation I don't where they got it

    vs.dco2star = co2calc_SWS(vs, t_in, s_in, dic_in, ta_in, co2_in, atmospheric_pressure)

    # Schmidt number for CO2
    # Wanninkhof, 1992, table A
    scco2 = 2073.1 - 125.62 * t_in + 3.6276 * t_in ** 2 - 0.043219 * t_in ** 3

    piston_vel = ao * xconv * (wind_speed) ** 2 * ((scco2/660.0)**(-0.5))
    # NOTE: According to https://www.soest.hawaii.edu/oceanography/courses/OCN623/Spring%202015/Gas_Exchange_2015_one-lecture1.pdf there are 3 regimes we are looking at the wavy surface
    # NOTE: https://aslopubs.onlinelibrary.wiley.com/doi/pdf/10.4319/lom.2014.12.351 uses the correct form for scco2
    # Sweeney et al. 2007

    # 1e3 added to convert to mmol / m^2 / s
    co2_flux = piston_vel * vs.dco2star * vs.maskT[:, :, -1] * 1e3
    # NOTE Is this Fick's First law? https://www.ocean.washington.edu/courses/oc400/Lecture_Notes/CHPT11.pdf

    # TODO set boundary condictions
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(co2_flux)

    # TODO land fluxes?

    # call co2forc
    return co2_flux


@veros_method
def oxygen_flux(vs):
    pass


@veros_method
def co2calc_SWS(vs, temperature, salinity, dic_in, ta_in, co2_in, atmospheric_pressure):
    """ Please rename this to something that actually makes sense
        Calculate delta co2* from total alkalinty and total CO2 at
        temperature, salinity and atmosphere total pressure
    """

    sit_in = np.ones_like(temperature) * 7.6875e-3  # mol / m^3 TODO: What is this?
    pt_in = np.ones_like(temperature) * 0.5125e-3  # mol / m^3 TODO: What is this?

    temperature_in_kelvin = temperature + 273.15

    # Change units from the input of mol/m^3 -> mol/kg:
    # where the ocean's mean surface density is 1024.5 kg/m^3
    # Note: mol/kg are actually what the body of this routine uses for calculations

    permil = 1.0 / 1024.5
    pt = pt_in * permil
    sit = sit_in * permil
    ta = ta_in * permil
    dic = dic_in * permil

    # convert from uatm to atm, but it's in ppmv
    permeg = 1e-6
    co2 = co2_in * permeg

    scl = salinity / 1.80655
    _is = 19.924 * salinity / (1000.0 - 1.005 * salinity)  # ionic _strength

    # Concentrations for borate, sulfate, and flouride
    bt = 0.000232 * scl / 10.811  # Uppstrom (1974)
    st = 0.14 * scl / 96.062  # Morris & Riley (1966)
    ft = 0.000067 * scl / 18.9984  # Riley (1965)

    # f = k0(1-pH20) * correction term for non-ideality
    # Weiss & Price (1980, Mar. Chem., 8, 347-359; Eq 13 with table 6 values)
    ff = np.exp(-162.8301 + 218.2968 / (temperature_in_kelvin / 100)
            + 90.9241 * np.log(temperature_in_kelvin / 100)
            - 1.47696 * (temperature_in_kelvin / 100)**2
            + salinity * (0.025695 - 0.025225 * temperature_in_kelvin / 100
            + 0.0049867 * (temperature_in_kelvin / 100)**2))

    # K0 from Weiss 1974
    k0 = np.exp(93.4517 / (temperature_in_kelvin / 100) - 60.2409 + 23.3585
            * np.log(temperature_in_kelvin / 100) + salinity * (
                0.023517 - 0.023656 * temperature_in_kelvin / 100 + 0.0047036 * (temperature_in_kelvin / 100) ** 2))

    # Now calculate FugFac according to Weiss (1974) Marine Chemestry TODO: What does FugFac mean?
    # delta_x and b_x are in cm3/mol TODO we should use m over cm right?
    # NOTE: This looks like equation 9, but without the pressure factors. Could FugFac mean fugacity factor??
    rt_x = 83.1451 * temperature_in_kelvin  # Gas constant times temperature
    delta_x = (57.7 - 0.118 * temperature_in_kelvin)  # In the text "\delta CO_2-air
    b_x = -1636.75 + 12.0408 * temperature_in_kelvin - 0.0327957 * temperature_in_kelvin ** 2 \
            + 3.16528 * 1e-5 * temperature_in_kelvin ** 3  # equation 6: Second viral coefficient B(T) (cm^3/mol)
    FugFac = np.exp((b_x + 2 * delta_x) / rt_x)  # equation 9 without front factor and ignoring pressure factors, but if they are 1atm, then that explains it


    # k1 = [H][HCO3]/[H2CO3]
    # k2 = [H][CO3]/[HCO3]     on hSWS TODO what does on hSWS mean?
    # Millero p.664 (1995) using Mehrbach et al. data on SEAWATER scale
    # (Original reference: Dickson and Millero, DSR, 1987)
    k1 = 10 ** (-1.0 * (3670.7 / temperature_in_kelvin - 62.088 + 9.7944 * np.log(temperature_in_kelvin) \
            - 0.0118 * salinity + 0.000116 * salinity**2))

    k2 = 10 ** (-1 * (1394.7 / temperature_in_kelvin + 4.777 - 0.0184 * salinity + 0.000118 * salinity**2))

    # k1p = [H][H2PO4]/[H3PO4] on hSWS TODO: hSWS?
    # Millero p.670 (1995)
    k1p = np.exp(-4576.752 / temperature_in_kelvin + 115.540 - 18.453 * np.log(temperature_in_kelvin) \
            + (-106.736 / temperature_in_kelvin + 0.69171) * np.sqrt(salinity) \
            + (-0.65643 / temperature_in_kelvin - 0.01844) * salinity)

    # k2p = [H][HPO3]/[H2PO4] on hSWS TODO:...
    k2p = np.exp(-8814.715 / temperature_in_kelvin + 172.1033 - 27.927 * np.log(temperature_in_kelvin) \
            + (-160.340 / temperature_in_kelvin + 1.3566) * np.sqrt(salinity) \
            + (0.37335 / temperature_in_kelvin - 0.05778) * salinity)


    # k3p = [H][PO4]/[HPO4] on hSWS TODO
    k3p = np.exp(-3070.75 / temperature_in_kelvin - 18.126 \
            + (17.27039 / temperature_in_kelvin + 2.81197) * np.sqrt(salinity) \
            + (-44.99486 / temperature_in_kelvin - 0.09984) * salinity)

    # ksi = [H][SiO(OH)3]/[Si(OH)4] on hSWS
    # Millero p.671 (1995) using data from Yao and Millero (1995)
    # change to (mol/ kg soln)
    # depth dependancy assumed to be the same as boric acid
    # typo in Millero 1994 corrected in sign of 0.1622
    ksi = np.exp(-8904.2 / temperature_in_kelvin + 117.400 - 19.334*np.log(temperature_in_kelvin) \
          + (-458.79 / temperature_in_kelvin + 3.5913) * np.sqrt(_is) \
          + (188.74 / temperature_in_kelvin - 1.5998) * _is \
          + (-12.1652 / temperature_in_kelvin + 0.07871) * _is**2 \
          + np.log(1.0 - 0.001005 * salinity))

    # kw = [H][OH] on hSWS
    # Millero p.670 (1995) using composite data
    # pressure dependancy in Millero 1994 corrected for sea water from
    # Millero 1983

    kw = np.exp(-13847.26 / temperature_in_kelvin + 148.9802 \
                - 23.6521 * np.log(temperature_in_kelvin) \
                + (118.67 / temperature_in_kelvin - 5.977 \
                + 1.0495 * np.log(temperature_in_kelvin)) * np.sqrt(salinity) - 0.01615 * salinity)

    # ks = [H][SO4]/[HSO4] on free H scale
    # Dickson (1990, J. chem. Thermodynamics 22, 113)
    # change to (mol/ kg soln)
    ks = np.exp(-4276.1 / temperature_in_kelvin + 141.328 - 23.093 * np.log(temperature_in_kelvin) \
       + (-13856 / temperature_in_kelvin + 324.57 - 47.986 * np.log(temperature_in_kelvin)) * np.sqrt(_is) \
       + (35474 / temperature_in_kelvin - 771.54 + 114.723 * np.log(temperature_in_kelvin)) * _is \
       - 2698 / temperature_in_kelvin * _is**1.5 + 1776 / temperature_in_kelvin * _is**2 \
       + np.log(1.0 - 0.001005 * salinity))

    # kf = [H][F]/[HF] on free H scale
    # Dickson and Riley (1979)
    # change to (mol/ kg soln)
    kf = np.exp(1590.2 / temperature_in_kelvin - 12.641 + 1.525 * np.sqrt(_is) + np.log(1.0 - 0.001005 * salinity))

    # kb = [H][BO2]/[HBO2] on hSWS
    # Dickson p.673 (1990)
    # change from htotal to hSWS
    # typo in Millero 1994 corrected in sign of 0.1622
    kb = np.exp((-8966.90 - 2890.53 * np.sqrt(salinity) - 77.942 * salinity \
       + 1.728 * salinity**1.5 - 0.0996 * salinity**2) / temperature_in_kelvin \
       + (148.0248 + 137.1942 * np.sqrt(salinity) + 1.62142 * salinity) \
       + (-24.4344 - 25.085 * np.sqrt(salinity) - 0.2474 * salinity) * np.log(temperature_in_kelvin) \
       + 0.053105 * np.sqrt(salinity) * temperature_in_kelvin \
       + np.log((1 + (st / ks) + (ft / kf)) / (1 + (st / ks))))

    # From UVic ESCM comments
    # Calculate [H+] SWS when DIC and TA are known at T, S and 1 atm.
    # The solution converges to err of xacc. The solution must be within
    # the range x1 to x2.
    #
    # If DIC and TA are known then either a root finding or iterative method
    # must be used to calculate hSWS. In this case we use the Newton-Raphson
    # "safe" method taken from "Numerical Recipes" (function "rtsafe.f" with
    # error trapping removed).
    #
    # As currently set, this procedure iterates about 12 times. The x1 and x2
    # values set below will accomodate ANY oceanographic values. If an initial
    # guess of the pH is known, then the number of iterations can be reduced to
    # about 5 by narrowing the gap between x1 and x2. It is recommended that
    # the first few time steps be run with x1 and x2 set as below. After that,
    # set x1 and x2 to the previous value of the pH +/- ~0.5. The current
    # setting of xacc will result in co2star accurate to 3 significant figures
    # (xx.y). Making xacc bigger will result in faster convergence also, but this
    # is not recommended (xacc of 10**-9 drops precision to 2 significant figures).

    ph = np.log10(vs.hSWS)  # negativ ph

    x1 = 10.0 ** (ph + 0.5)
    x2 = 10.0 ** (ph - 0.5)
    xacc = 1e-10

    in1 = np.ones_like(co2_in) * x1
    in2 = np.ones_like(co2_in) * x2

    iter_mask = np.empty_like(in1, dtype=np.bool)
    iter_mask[...] = vs.maskT[:, :, -1]

    # Find hSWS by root finding algorithm
    vs.hSWS = drtsafe(vs, iter_mask, ta_iter_SWS, in1, in2, args=(k1, k2, k1p, k2p, k3p, st, ks, kf, ft, dic, ta, sit, ksi, pt, bt, kw, kb), accuracy=xacc)

    # Calculate [CO2*] as defined in DOE Methods Handbook 1993 Ver. 2,
    # ORNL/CDIC-74, Dickson and Goyet, eds. (Ch 2 p 1+, Eq A.49)
    co2star = dic * vs.hSWS**2 / (vs.hSWS**2 + k1 * vs.hSWS + k1 * k2)
    co2starair = co2 * ff * atmospheric_pressure
    dco2star = co2starair - co2star

    pCO2 = co2star / (k0 * FugFac)
    dpCO2 = pCO2 - co2 * atmospheric_pressure

    # Convert units back
    vs.co2star[:, :] = co2star / permil
    vs.dco2star[:, :] = dco2star / permil

    vs.pCO2[:, :] = pCO2 / permeg
    vs.dpCO2[:, :] = dpCO2 / permeg

    return vs.dco2star


@veros_method
def ta_iter_SWS_numpy(vs, x, k1, k2, k1p, k2p, k3p, st, ks, kf, ft, dic, ta, sit, ksi, pt, bt, kw, kb):
    """
    Sum of free protons calculated from saturation constants
    """
    x2 = x * x
    x3 = x2 * x
    k12 = k1 * k2
    k12p = k1p * k2p
    k123p = k12p * k3p
    c = 1.0 + st / ks + ft / kf
    a = x3 + k1p * x2 + k12p * x + k123p
    a2 = a * a
    da = 3.0 * x2 + 2.0 * k1p * x + k12p
    b = x2 + k1 * x + k12
    b2 = b * b
    db = 2.0 * x + k1

    # fn = hco3+co3+borate+oh+hpo4+2*po4+silicate-hfree-hso4-hf-h3po4-ta
    fn = k1 * x * dic / b + 2.0 * dic * k12 / b + \
         bt / (1.0 + x / kb) + kw / x + pt * k12p * x / a + \
         2.0 * pt * k123p / a + sit / (1.0 + x / ksi) - x / c - \
         st / (1.0 + ks / (x / c)) - ft / (1.0 + kf / (x / c)) -\
         pt * x3/a - ta

    # df = dfn/dx
    df = ((k1 * dic * b) - k1 * x * dic * db) / b2 - 2.0 * dic * k12 * db / b2 - \
         bt / kb / (1.0 + x / kb)**2.0 - kw / x2 + (pt * k12p * (a - x * da)) / a2 - \
         2.0 * pt * k123p * da / a2 - sit / ksi / (1.0 + x / ksi)**2 - 1.0 / c - \
         st * (1.0 + ks / (x / c))**(-2.0) * (ks * c / x2) - ft * (1.0 + kf / (x / c))**(-2.0) * \
         (kf * c / x2) - pt * x2 * (3.0 * a - x * da) / a2

    return fn, df


@veros_method
def ta_iter_SWS_bohrium(*args):
    """
    Sum of free proton calculated from saturation constants
    For bohrium, making use of a mask to limit the number of calculations
    """

    arg_behaving = [np.user_kernel.make_behaving(args[1], dtype=np.int32)]

    for i, arg in enumerate(args[2:]):
        arg_behaving.append(np.user_kernel.make_behaving(arg, dtype=np.double))

    mask, x, k1, k2, k1p, k2p, k3p, st, ks, kf, ft, dic, ta, sit, ksi, pt, bt, kw, kb = arg_behaving

    fn = np.empty_like(x)
    df = np.empty_like(x)

    kernel = """
    #include <stdint.h>
    #include <stdlib.h>

    void execute(int *mask, double *x, double *k1, double *k2, double *k1p, double *k2p, double *k3p, double *st, double *ks, double *kf, double *ft, double *dic, double *ta, double *sit, double *ksi, double *pt, double *bt, double *kw, double *kb, double *fn, double *df) {

        for (uint64_t i=0; i<%(shape)d; ++i) {
            if (!mask[i]) {
                continue;
            }
            double x2 = x[i] * x[i];
            double x3 = x2 * x[i];
            double k12 = k1[i] * k2[i];
            double k12p = k1p[i] * k2p[i];
            double k123p = k12p * k3p[i];
            double c = 1.0 + st[i] / ks[i] + ft[i] / kf[i];
            double a = x3 + k1p[i] * x2 + k12p * x[i] + k123p;
            double a2 = a * a;
            double da = 3.0 * x2 + 2.0 * k1p[i] * x[i] + k12p;
            double b = x2 + k1[i] * x[i] + k12;
            double b2 = b * b;
            double db = 2.0 * x[i] + k1[i];


            fn[i] = k1[i] * x[i] * dic[i] / b
                    + 2.0 * dic[i] * k12 / b
                    + bt[i] / (1.0 + x[i] / kb[i])
                    + kw[i] / x[i]
                    + pt[i] * k12p * x[i] / a
                    + 2.0 * pt[i] * k123p / a
                    + sit[i] / (1.0 + x[i] / ksi[i])
                    - x[i] / c
                    - st[i] / (1.0 + ks[i] / (x[i] / c))
                    - ft[i] / (1.0 + kf[i] / (x[i] / c))
                    - pt[i] * x3 / a - ta[i];

            double t1 = (1.0 + x[i] / kb[i]);
            double t2 = (1.0 + x[i] / ksi[i]);
            double t3 = (1.0 + ks[i] / (x[i] / c));
            double t4 = (1.0 + kf[i] / (x[i] / c));

            df[i] = (k1[i] * dic[i] * b - k1[i] * x[i] * dic[i] * db) / b2
                    - 2.0 * dic[i] * k12 * db / b2
                    - bt[i] / kb[i] / (t1 * t1)
                    - kw[i] / x2
                    + (pt[i] * k12p * (a - x[i] * da)) / a2
                    - 2.0 * pt[i] * k123p * da / a2
                    - sit[i] / ksi[i] / (t2 * t2)
                    - 1.0 / c
                    - st[i] / (t3 * t3) * ks[i] * c / x2
                    - ft[i] / (t4 * t4) * kf[i] * c / x2
                    - pt[i] * x2 * (3.0 * a - x[i] * da) / a2;


        }
    }

    """ % {"shape": x.size}

    np.user_kernel.execute(kernel, arg_behaving + [fn, df])

    return fn, df


@veros_method
def ta_iter_SWS(*args):
    """
    Function to be optimized for free protons, calls Bohrium version
    if required
    """
    vs = args[0]
    mask = args[1]

    if vs.backend_name == "bohrium":
        return ta_iter_SWS_bohrium(vs, mask, *args[2:])
    else:
        return ta_iter_SWS_numpy(vs, *args[2:])


@veros_method
def drtsafe_boundary_update(vs, mask, x_low, x_high, df, f, f_low, f_high, drtsafe_val):
    """
    For drtsafe: Moves the search boundary based on current result.
    Masked to avoid syncs to numpy
    """

    if vs.backend_name == "bohrium":
        mask_input = np.user_kernel.make_behaving(mask, dtype=np.bool)
        x_low_input = np.user_kernel.make_behaving(x_low, dtype=vs.default_float_type)
        x_high_input = np.user_kernel.make_behaving(x_high, dtype=vs.default_float_type)
        df_input = np.user_kernel.make_behaving(df, dtype=vs.default_float_type)
        f_input = np.user_kernel.make_behaving(f, dtype=vs.default_float_type)
        f_low_input = np.user_kernel.make_behaving(f_low, dtype=vs.default_float_type)
        f_high_input = np.user_kernel.make_behaving(f_high, dtype=vs.default_float_type)
        drtsafe_val_input = np.user_kernel.make_behaving(drtsafe_val, dtype=vs.default_float_type)

        kernel = """
        #include <stdint.h>
        #include <stdlib.h>
        #include <stdbool.h>
        void execute(bool *mask, double *x_low, double *x_high, double *df, double *f, double *f_low, double *f_high, double *drtsafe_val) {
            for (uint64_t i=0; i<%(shape0)d; ++i) {

                    if (!mask[i]) {
                        continue;
                    }
                    if (f[i] < 0.0) {
                        x_low[i] = drtsafe_val[i];
                        f_low[i] = f[i];
                    } else {
                        x_high[i] = drtsafe_val[i];
                        f_high[i] = f[i];
                    }
            }
        }
        """ % {'shape0': mask.shape[0] * mask.shape[1]}
        np.user_kernel.execute(kernel, [mask_input, x_low_input, x_high_input, df_input, f_input, f_low_input, f_high_input, drtsafe_val_input])

        # copy result back into variables
        x_low[...] = x_low_input[...]
        f_low[...] = f_low_input[...]
        x_high[...] = x_high_input[...]
        f_high[...] = f_high_input[...]

    else:
        x_low[mask], x_high[mask], f_low[mask], f_high[mask] = np.where(f[mask] < 0,
                (drtsafe_val[mask], x_high[mask], f[mask], f_high[mask]),
                (x_low[mask], drtsafe_val[mask], f_low[mask], f[mask]))


@veros_method
def drtsafe_step(vs, mask, drtsafe_val, x_high, df, f, x_low, dx, dx_old, accuracy):
    """
    Update step size: If step size would take us out of bounds,
    set step size to half the interval length, else f/df
    and update step accordingly
    """

    if vs.backend_name == "bohrium":
        mask_input = np.user_kernel.make_behaving(mask, dtype=np.int32)
        drtsafe_val_input = np.user_kernel.make_behaving(drtsafe_val, dtype=vs.default_float_type)
        x_high_input = np.user_kernel.make_behaving(x_high, dtype=vs.default_float_type)
        df_input = np.user_kernel.make_behaving(df, dtype=vs.default_float_type)
        f_input = np.user_kernel.make_behaving(f, dtype=vs.default_float_type)
        x_low_input = np.user_kernel.make_behaving(x_low, dtype=vs.default_float_type)
        dx_input = np.user_kernel.make_behaving(dx, dtype=vs.default_float_type)
        dx_old_input = np.user_kernel.make_behaving(dx_old, dtype=vs.default_float_type)

        kernel = """
        #include <stdint.h>
        #include <stdlib.h>
        #include <stdbool.h>
        #include <math.h>
        void execute(int *mask, double *drtsafe_val, double *x_high, double *df, double *f, double *x_low, double *dx, double *dx_old) {
            for (uint64_t i=0; i<%(shape0)d; ++i) {
                if (!mask[i]) {
                    continue;
                }

                double step1 = (drtsafe_val[i] - x_high[i]) * df[i] - f[i];
                double step2 = (drtsafe_val[i] - x_low[i]) * df[i] - f[i]; 
                bool step_inside = (step1 * step2) >= 0;
                bool tmp_mask = step_inside || (abs(2.0 * f[i]) > abs(dx_old[i] * df[i]));

                dx_old[i] = dx[i];
                if (tmp_mask) {
                    dx[i] = 0.5 * (x_high[i] - x_low[i]);
                    drtsafe_val[i] = x_low[i] + dx[i];
                } else {
                    dx[i] = f[i] / df[i];
                    drtsafe_val[i] = drtsafe_val[i] - dx[i];
                }
                mask[i] = fabs(dx[i]) > %(accuracy)E;
            }
        }
        """ % {'shape0': mask.size, 'accuracy': accuracy}
        np.user_kernel.execute(kernel, [mask_input, drtsafe_val_input, x_high_input, df_input, f_input, x_low_input, dx_input, dx_old_input])

        # Copy results back out
        mask[...] = mask_input[...]
        drtsafe_val[...] = drtsafe_val_input[...]
        dx[...] = dx_input[...]
        dx_old[...] = dx_old_input[...]

    else:
        tmp_mask = ((drtsafe_val[mask] - x_high[mask]) * df[mask] - f[mask]) * (
                    (drtsafe_val[mask] - x_low[mask]) * df[mask] - f[mask]) >= 0
        tmp_mask = np.logical_or(tmp_mask, (np.abs(2.0 * f[mask]) > np.abs(dx_old[mask] * df[mask])))

        dx_old[:] = dx.copy()
        dx[mask] = np.where(tmp_mask, 0.5 * (x_high[mask] - x_low[mask]), f[mask] / df[mask])
        drtsafe_val[mask] = np.where(tmp_mask, x_low[mask] + dx[mask], drtsafe_val[mask] - dx[mask])

        mask[mask] = np.abs(dx[mask]) > accuracy


@veros_method
def drtsafe(vs, mask, function, guess_low, guess_high, args=None, accuracy=1e-10, max_iterations=100):
    """
    Masked version of drtsafe
    the function given should return a function value and derivative.
    It will be given the updated mask as first argument and guess as second.
    Any further arguments will be what is stored in args.
    The initial guess will be the mean value of guess_low and guess_high
    """

    # Initial guess and step size
    drtsafe_val = 0.5 * (guess_low + guess_high)
    dx = np.abs(guess_high - guess_low)
    dx_old = dx.copy()

    # Function value at boundaries and guess
    f_low, _ = function(vs, mask, guess_low, *args)
    f_high, _ = function(vs, mask, guess_high, *args)
    f, df = function(vs, mask, drtsafe_val, *args)

    # Ensure low values have negative function value
    # and high has positive
    x_low, x_high, f_low, f_high = np.where(f_low < 0,
            (guess_low, guess_high, f_low, f_high),
            (guess_high, guess_low, f_high, f_low))

    for _ in range(max_iterations):

        # update step size, step and mask
        drtsafe_step(vs, mask, drtsafe_val, x_high, df, f, x_low, dx, dx_old, accuracy)

        if not mask.any():  # complete
            break

        # Update function for next step
        f, df = function(vs, mask, drtsafe_val, *args)

        # Update search boundaries
        drtsafe_boundary_update(vs, mask, x_low, x_high, df, f, f_low, f_high, drtsafe_val)

    return drtsafe_val

