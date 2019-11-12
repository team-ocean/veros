"""
Functions calculating atmosphere-ocean fluxes

"""
from veros import veros_method, runtime_settings as rs
from . import utilities


@veros_method
def carbon_flux(vs):
    """Calculates flux of CO2 over the ocean-atmosphere boundary

    This is an adaptation of co2_calc_SWS from UVic ESCM

    Note
    ----
    This was written without an atmosphere component in veros.
    Therefore an atmospheric pressure of 1 atm is assumed.
    The concentration of CO2 (in units of ppmv) may be set in vs.atmospheric_co2

    Note
    ----
    This was written without an explicit sea ice component. Therefore a full
    ice cover is assumed when temperature is below -1.8C and temperature forcing is negative

    Returns
    -------
    numpy.ndarray(vs.nx, vs.ny) with flux in units of :math:`mmol / m^2 / s`
    Positive indicates a flux into the ocean
    """

    icemask = np.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
            vs.forc_temp_surface < 0.0)
    ao = np.logical_not(icemask)

    atmospheric_pressure = 1  # atm  NOTE: We don't have an atmosphere yet, hence constant pressure

    # TODO get actual wind speed rather than deriving from wind stress
    wind_speed = np.sqrt(np.abs(vs.surface_taux / vs.rho_0) + np.abs(vs.surface_tauy / vs.rho_0)) * 500
    vs.wind_speed = wind_speed


    # xconv is constant to convert piston_vel from cm/hr -> cm/s
    # here it is 100.*a*xconv (100 => m to cm, a=0.337, xconv=1/3.6e+05)
    xconv = 0.337 / 3.6e5
    xconv *= 0.75  # NOTE: This seems like an approximation I don't know where they got it

    vs.dco2star = co2calc_SWS(vs, vs.temp[:, :, -1, vs.tau],  # [degree C]
                              vs.salt[:, :, -1, vs.tau],  # [g/kg]
                              vs.dic[:, :, -1, vs.tau] * 1e-3,  # [mmol -> mol]
                              vs.alkalinity[:, :, -1, vs.tau] * 1e-3,  # [mmol -> mol]
                              vs.atmospheric_co2,  # [ppmv]
                              atmospheric_pressure)  # atm

    # Schmidt number for CO2
    # Wanninkhof, 1992, table A
    scco2 = 2073.1 - 125.62 * vs.temp[:, :, -1, vs.tau] + 3.6276 * vs.temp[:, :, -1, vs.tau] ** 2 - 0.043219 * vs.temp[:, :, -1, vs.tau] ** 3

    piston_vel = ao * xconv * (wind_speed) ** 2 * ((scco2/660.0)**(-0.5))
    # NOTE: According to https://www.soest.hawaii.edu/oceanography/courses/OCN623/Spring%202015/Gas_Exchange_2015_one-lecture1.pdf there are 3 regimes we are looking at the wavy surface
    # NOTE: https://aslopubs.onlinelibrary.wiley.com/doi/pdf/10.4319/lom.2014.12.351 uses the correct form for scco2
    # Sweeney et al. 2007

    # 1e3 added to convert to mmol / m^2 / s
    co2_flux = piston_vel * vs.dco2star * vs.maskT[:, :, -1] * 1e3

    utilities.enforce_boundaries(vs, co2_flux)

    vs.cflux[...] = co2_flux
    return vs.cflux


@veros_method
def co2calc_SWS(vs, temperature, salinity, dic_in, ta_in, co2_in, atmospheric_pressure):
    """ Calculate delta co2*

        Calculate delta co2* from total alkalinty and total CO2 at temperature, salinity and atmosphere total pressure
        This function needs a better name

        Parameters
        ----------

        temperature
            sea surface temperature

        salinity
            sea surface salinity

        dic_in
            DIC in surface layer in units [mol/m^3]

        ta_in
            Total alkalinity in surface layer in units [mol/m^3]

        co2_in
            atmospheric co2

        atmospheric_pressure
            atmospheric pressure in [atm]

        Note
        ----
        It is assumed, that total Si and P are constant.
        Several other element concentration are approximated from temperature and salinity

        Returns
        -------

        $\delta CO_2\star$
    """

    sit_in = np.ones_like(temperature) * 7.6875e-3  # [mol/m^3] estimated total Si
    pt_in = np.ones_like(temperature) * 0.5125e-3  # [mol/m^3 ] estimated total P

    temperature_in_kelvin = temperature + 273.15

    # Change units from the input of mol/m^3 -> mol/kg:
    # where the ocean's mean surface density is 1024.5 kg/m^3
    # Note: mol/kg are actually what the body of this routine uses for calculations

    permil = 1.0 / 1024.5
    pt = pt_in * permil  # total P
    sit = sit_in * permil  # total Si
    ta = ta_in * permil  # total alkalinity
    dic = dic_in * permil

    # convert from uatm to atm, but it's in ppmv
    permeg = 1e-6
    co2 = co2_in * permeg

    scl = salinity / 1.80655  # convenience parameter
    _is = 19.924 * salinity / (1000.0 - 1.005 * salinity)  # ionic _strength

    # Concentrations for borate, sulfate, and flouride
    bt = 0.000232 * scl / 10.811  # Uppstrom (1974)
    st = 0.14 * scl / 96.062  # Morris & Riley (1966)
    ft = 0.000067 * scl / 18.9984  # Riley (1965)

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

    # Now calculate FugFac according to Weiss (1974) Marine Chemestry
    rt_x = 83.1451 * temperature_in_kelvin  # Gas constant times temperature
    delta_x = (57.7 - 0.118 * temperature_in_kelvin)  # In the text '\delta CO_2-air
    b_x = -1636.75 + 12.0408 * temperature_in_kelvin - 0.0327957 * temperature_in_kelvin ** 2 \
            + 3.16528 * 1e-5 * temperature_in_kelvin ** 3  # equation 6: Second viral coefficient B(T) (cm^3/mol)
    FugFac = np.exp((b_x + 2 * delta_x) / rt_x)  # equation 9 without front factor and ignoring pressure factors


    # k1 = [H][HCO3]/[H2CO3]
    # k2 = [H][CO3]/[HCO3]
    # Millero p.664 (1995) using Mehrbach et al. data on SEAWATER scale
    # (Original reference: Dickson and Millero, DSR, 1987)
    k1 = 10 ** (-1.0 * (3670.7 / temperature_in_kelvin - 62.088 + 9.7944 * np.log(temperature_in_kelvin)
                - 0.0118 * salinity + 0.000116 * salinity**2))

    k2 = 10 ** (-1 * (1394.7 / temperature_in_kelvin + 4.777 - 0.0184 * salinity + 0.000118 * salinity**2))

    # k1p = [H][H2PO4]/[H3PO4]
    # Millero p.670 (1995)
    k1p = np.exp(-4576.752 / temperature_in_kelvin + 115.540 - 18.453 * np.log(temperature_in_kelvin)
                 + (-106.736 / temperature_in_kelvin + 0.69171) * np.sqrt(salinity)
                 + (-0.65643 / temperature_in_kelvin - 0.01844) * salinity)

    # k2p = [H][HPO3]/[H2PO4]
    k2p = np.exp(-8814.715 / temperature_in_kelvin + 172.1033 - 27.927 * np.log(temperature_in_kelvin)
                 + (-160.340 / temperature_in_kelvin + 1.3566) * np.sqrt(salinity)
                 + (0.37335 / temperature_in_kelvin - 0.05778) * salinity)

    # k3p = [H][PO4]/[HPO4]
    k3p = np.exp(-3070.75 / temperature_in_kelvin - 18.126
                 + (17.27039 / temperature_in_kelvin + 2.81197) * np.sqrt(salinity)
                 + (-44.99486 / temperature_in_kelvin - 0.09984) * salinity)

    # ksi = [H][SiO(OH)3]/[Si(OH)4]
    # Millero p.671 (1995) using data from Yao and Millero (1995)
    # change to (mol/ kg soln)
    # depth dependancy assumed to be the same as boric acid
    # typo in Millero 1994 corrected in sign of 0.1622
    ksi = np.exp(-8904.2 / temperature_in_kelvin + 117.400 - 19.334*np.log(temperature_in_kelvin)
                 + (-458.79 / temperature_in_kelvin + 3.5913) * np.sqrt(_is)
                 + (188.74 / temperature_in_kelvin - 1.5998) * _is
                 + (-12.1652 / temperature_in_kelvin + 0.07871) * _is**2
                 + np.log(1.0 - 0.001005 * salinity))

    # kw = [H][OH]
    # Millero p.670 (1995) using composite data
    # pressure dependancy in Millero 1994 corrected for sea water from
    # Millero 1983

    kw = np.exp(-13847.26 / temperature_in_kelvin + 148.9802
                - 23.6521 * np.log(temperature_in_kelvin)
                + (118.67 / temperature_in_kelvin - 5.977
                   + 1.0495 * np.log(temperature_in_kelvin)) * np.sqrt(salinity) - 0.01615 * salinity)

    # ks = [H][SO4]/[HSO4] on free H scale
    # Dickson (1990, J. chem. Thermodynamics 22, 113)
    # change to (mol/ kg soln)
    ks = np.exp(-4276.1 / temperature_in_kelvin + 141.328 - 23.093 * np.log(temperature_in_kelvin)
                + (-13856 / temperature_in_kelvin + 324.57 - 47.986 * np.log(temperature_in_kelvin)) * np.sqrt(_is)
                + (35474 / temperature_in_kelvin - 771.54 + 114.723 * np.log(temperature_in_kelvin)) * _is
                - 2698 / temperature_in_kelvin * _is**1.5 + 1776 / temperature_in_kelvin * _is**2
                + np.log(1.0 - 0.001005 * salinity))

    # kf = [H][F]/[HF] on free H scale
    # Dickson and Riley (1979)
    # change to (mol/ kg soln)
    kf = np.exp(1590.2 / temperature_in_kelvin - 12.641 + 1.525 * np.sqrt(_is) + np.log(1.0 - 0.001005 * salinity))

    # kb = [H][BO2]/[HBO2]
    # Dickson p.673 (1990)
    # change from htotal to hSWS
    # typo in Millero 1994 corrected in sign of 0.1622
    kb = np.exp((-8966.90 - 2890.53 * np.sqrt(salinity) - 77.942 * salinity
                 + 1.728 * salinity**1.5 - 0.0996 * salinity**2) / temperature_in_kelvin
                 + (148.0248 + 137.1942 * np.sqrt(salinity) + 1.62142 * salinity)
                 + (-24.4344 - 25.085 * np.sqrt(salinity) - 0.2474 * salinity) * np.log(temperature_in_kelvin)
                 + 0.053105 * np.sqrt(salinity) * temperature_in_kelvin
                 + np.log((1 + (st / ks) + (ft / kf)) / (1 + (st / ks))))

    # From UVic ESCM comments
    # Calculate [H+] SWS when DIC and TA are known at T, S and 1 atm.
    # The solution converges to err of xacc. The solution must be within
    # the range x1 to x2.
    #
    # If DIC and TA are known then either a root finding or iterative method
    # must be used to calculate hSWS. In this case we use the Newton-Raphson
    # 'safe' method taken from 'Numerical Recipes' (function 'rtsafe.f' with
    # error trapping removed).

    xacc = 1e-10

    # NOTE hSWS should never exceed the safe values, but we can check it
    limit_min = 1e-8
    boundary = np.maximum(limit_min,
                          np.minimum(1e-6 - vs.hSWS, vs.hSWS - 1e-10))

    # boundary guesses are placed equidistant from the guessed best solution
    in1 = vs.hSWS + boundary  # input guess 1
    in2 = vs.hSWS - boundary  # input guess 2

    # where to run the optimization - skip land
    iter_mask = np.empty_like(in1, dtype=np.bool)
    iter_mask[...] = vs.maskT[:, :, -1]

    # Find hSWS by root finding algorithm
    vs.hSWS = drtsafe_masked(vs, iter_mask, ta_iter_SWS, in1, in2,
                             args=(k1, k2, k1p, k2p, k3p, st, ks, kf, ft,
                                   dic, ta, sit, ksi, pt, bt, kw, kb), accuracy=xacc)

    # Calculate [CO2*] as defined in DOE Methods Handbook 1993 Ver. 2,
    # ORNL/CDIC-74, Dickson and Goyet, eds. (Ch 2 p 1+, Eq A.49)
    co2star = dic * vs.hSWS**2 / (vs.hSWS**2 + k1 * vs.hSWS + k1 * k2)
    co2starair = co2 * ff * atmospheric_pressure
    dco2star = co2starair - co2star  # effective CO2 pressure difference

    # pCO2 and dpCO2 are for diagnostic purposes only
    pCO2 = co2star / (k0 * FugFac)  # partial CO2 pressure
    dpCO2 = pCO2 - co2 * atmospheric_pressure  # difference in CO2 pressure between ocean and atmosphere

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

    Parameters are same as ta_iter_SWS

    Note
    ----
    This function is internal and is intended to be called by ta_iter_SWS
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

    Parameters are same as ta_iter_SWS

    Note
    ----
    This function is internal. It was made explicitly for bohrium, making use of a mask to limit the number of calculations
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

    """ % {'shape': x.size}

    np.user_kernel.execute(kernel, arg_behaving + [fn, df])

    return fn, df


@veros_method
def ta_iter_SWS(*args):
    """
    Function to be optimized for free protons, calls Bohrium version if required


    Parameters
    ----------
    mask
        Mask indicating which cells to include / exclude from calculation
    x
        Free protons
    k1
    k2
    k1p
    k2p
    k3p
    st
    ks
    kf
    ft
    dic
    ta
    sit
    ksi
    pt
    bt
    kw
    kb

    Returns
    -------
    A function value and derivative in a tuple (f, df)
    """
    vs = args[0]
    mask = args[1]

    if rs.backend == 'bohrium':
        return ta_iter_SWS_bohrium(vs, mask, *args[2:])
    else:
        return ta_iter_SWS_numpy(vs, *args[2:])


@veros_method
def drtsafe_boundary_update(vs, mask, x_low, x_high, df, f, f_low, f_high, drtsafe_val):
    """
    For drtsafe: Moves the search boundary based on current result.

    Parameters
    ----------
    mask
        Mask for where to perform / skip calculations

    x_low
        Lower boundary value

    x_high
        Upper boundary value

    df
        Function derivative

    f
        Function value

    f_low
        Function value at lower bound

    f_high
        Function value at upper bound

    drtsafe_val
        Values for which f was calculated

    Returns
    -------
    None, but x_low, f_low, x_high, and f_high are updated to reflect new boundaries

    Note
    ----
    With the Bohrium backend, this function uses a user kernel to handle masked
    arrays. This is siginificantly faster than syncing to numpy or working on
    already completed cells.
    """

    if rs.backend == 'bohrium':
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
    Update step size

    If step size would take us out of bounds, set step size to half the interval length, else f/df
    and update step accordingly

    Parameters
    ----------
    mask
        Which cells should be calculated / skipped

    drtsafe_val
        Function input values

    x_high
        Upper bound of input values

    df
        Derivative in f

    f
        Function value

    x_low
        Lower bound of input values

    dx
        Change in function input value

    dx_old
        Change in function input value from previous step

    accuracy
        If calculated dx falls below this value, the mask for that cell will be updated

    Returns
    -------
    None, but mask, dtrsafe_val, dx, and dx_old will be updated


    Note
    ----
    With the Bohrium backend, this function uses a user kernel to skip completed cells.
    This is siginificantly faster than syncing to numpy or working on already completed cells.
    """

    if rs.backend == 'bohrium':
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
        np.user_kernel.execute(kernel, [mask_input, drtsafe_val_input, x_high_input, df_input,
                                        f_input, x_low_input, dx_input, dx_old_input])

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
def drtsafe_masked(vs, mask, function, guess_low, guess_high, args=None,
                   accuracy=1e-10, max_iterations=100):
    """
    Masked version of drtsafe

    Parameters
    ----------
    mask
        A mask indicating which cells may be skipped from calculations

    function
        Function to be optimzed. It must return a tuple consiting of function value and derivative

    guess_low
        Lower bound of initial guess bouding interval

    guess_high
        Upper bound of initial guess bounding interval

    args
        Additional fixed arguments passed to the function

    accuracy
        Maximum step size required as stop condition

    max_iterations
        Maximum number of optimization steps

    Returns
    -------
    Parameter values minimizing the function value
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


@veros_method
def drtsafe(vs, function, guess_low, guess_high, args=None, accuracy=1e-10, max_iterations=100):
    """ Root finding method with bounding box

    Follows Newton-Raphson step unless unsufficient or the step with take the solution out of bounds.
    Otherwise does birfurcating step.

    Parameters
    ----------
    function
        Function to be optimzed. It must return a tuple consiting of function value and derivative.

    guess_low
        Lower bound of initial guess bouding interval

    guess_high
        Upper bound of initial guess bounding interval

    args
        Additional fixed arguments passed to the function

    accuracy
        Maximum step size required as stop condition

    max_iterations
        Maximum number of optimization steps

    Returns
    -------
    Parameter values minimizing the function value
    """
    # Initial guess and step size
    drtsafe_val = 0.5 * (guess_low + guess_high)
    dx = np.abs(guess_high - guess_low)
    dx_old = dx.copy()

    # Function value at boundaries and guess
    f_low, _ = function(vs, guess_low, *args)
    f_high, _ = function(vs, guess_high, *args)
    f, df = function(vs, drtsafe_val, *args)

    # Ensure low values have negative function value
    # and high has positive
    x_low, x_high, f_low, f_high = np.where(f_low < 0,
                                            (guess_low, guess_high, f_low, f_high),
                                            (guess_high, guess_low, f_high, f_low))

    for _ in range(max_iterations):

        # update step size, step and mask
        step_mask = ((drtsafe_val - x_high) * df - f) * (
                    (drtsafe_val - x_low) * df - f) >= 0
        step_mask = np.logical_or(step_mask, (np.abs(2.0 * f) > np.abs(dx_old * df)))

        dx_old[:] = dx.copy()
        dx = np.where(step_mask, 0.5 * (x_high - x_low), f / df)

        if not (np.abs(dx) > accuracy).any():
            break

        # NOTE this was above the accuracy check
        drtsafe_val = np.where(step_mask, x_low + dx, drtsafe_val - dx)

        # Update function for next step
        f, df = function(vs, drtsafe_val, *args)

        # Update search boundaries
        x_low, x_high, f_low, f_high = np.where(f < 0,
                                                (drtsafe_val, x_high, f, f_high),
                                                (x_low, drtsafe_val, f_low, f))

    return drtsafe_val
