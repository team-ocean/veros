"""
Functions calculating atmosphere-ocean fluxes

"""
from veros import veros_method
import numpy as np  # TODO remove this as veros handles it
from scipy import optimize
from . import cyclic


@veros_method
def carbon_flux(vs):
    t_in = vs.temp[:, :, -1, vs.tau]  # [degree C] TODO rename variable
    s_in = vs.salt[:, :, -1, vs.tau]  # [g/kg = PSU] TODO rename variable
    dic_in = vs.dic[:, :, -1] #* 1e-3# [mmol -> mol] TODO rename variable
    ta_in = vs.alkalinity[:, :, -1] #* 1e-3 # [mmol -> mol] TODO rename variable
    co2_in = vs.atmospheric_co2 # [ppmv] TODO rename variable

    ao = 1  # 1 - ice fraction coverage
    # icemask = np.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
    #         vs.forc_temp_surface < 0.0)
    # ao = np.logical_not(icemask)


    atmospheric_pressure = 1  # atm  NOTE: We don't have an atmosphere yet, hence constant pressure

    # sbc(i, j, isw) = surface wind speed direction is specified by angle
    # TODO get actual wind speed rather than deriving from wind stress
    wind_speed = np.sqrt(np.abs(vs.surface_taux) + np.abs(vs.surface_tauy)) * 500 # mult for scale?
    vs.wind_speed = wind_speed


# !     xconv is constant to convert piston_vel from cm/hr -> cm/s
# !     here it is 100.*a*xconv (100 => m to cm, a=0.337, xconv=1/3.6e+05)
#       xconv = 33.7/3.6e+05
#       xconv = xconv*0.75
    xconv = 0.337 / 3.6e5
    xconv *= 0.75

    # co2star, dco2star = co2calc_SWS(vs, t_in, s_in, dic_in, ta_in, co2_in, atmospheric_pressure)
    dco2star = co2calc_SWS(vs, t_in, s_in, dic_in, ta_in, co2_in, atmospheric_pressure)
    vs.dco2star = dco2star

    # Schmidt number for CO2
    # t_in wasn't actually used but sst was, however they are the same...
    scco2 = 2073.1 - 125.62 * t_in + 3.6276 * t_in ** 2 - 0.043219 * t_in ** 3

    # piston_vel = ao * xconv * ((sbc[i, j, iws]) * 0.01) ** 2 * ((scco2/660.0)**(-0.5))
    # piston_vel = ao * xconv * (wind_speed * 0.01) ** 2 * ((scco2/660.0)**(-0.5))
    # NOTE units: m * (m / s)^2
    piston_vel = ao * xconv * (wind_speed) ** 2 * ((scco2/660.0)**(-0.5))
    # 1e3 added to convert to mmol / m^2 / s
    DIC_Flux = piston_vel * dco2star * vs.maskT[:, :, -1] # * 1e3

    # TODO set boundary condictions
    if vs.enable_cyclic_x:
        cyclic.setcyclic_x(DIC_Flux)

    # TODO land fluxes?

    # call co2forc
    return DIC_Flux

@veros_method
def oxygen_flux(vs):
    pass

@veros_method
def co2calc_SWS(vs, temperature, salinity, dic_in, ta_in, co2_in, atmospheric_pressure):
    """ Please rename this to something that actually makes sense
        Calculate delta co2* from total alkalinty and total CO2 at
        temperature, salinity and atmosphere total pressure
    """

    # Hardwire constants
    ph_high = 6.0
    ph_low = 10.0
    sit_in = 7.6875e-3  # mol / m^3 TODO: What is this?
    pt_in = 0.5125e-3  # mol / m^3  TODO: What is this?

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
    _is = 19.924 * salinity / (1000.0 - 1.005 * salinity)  # TODO What is this supposed to be?

    # Concentrations for borate, sulfate, and flouride
    bt = 0.000232 * scl / 10.811 # Uppstrom (1974)
    st = 0.14 * scl / 96.062  # Morris & Riley (1966)
    ft = 0.000067 * scl / 18.9984  # Riley (1965)

    # f = k0(1-pH20) * correction term for non-ideality
    # Weiss & Price (1980, Mar. Chem., 8, 347-359; Eq 13 with table 6 values)
    ff = np.exp(-162.8301 + 218.2968 / (temperature_in_kelvin / 100) \
            + 90.9241 * np.log(temperature_in_kelvin / 100) \
            - 1.47696 * (temperature_in_kelvin / 100)**2 \
            + salinity * (0.025695 - 0.025225 * temperature_in_kelvin / 100 \
            + 0.0049867 * (temperature_in_kelvin / 100)**2))

    # K0 from Weiss 1974
    k0 = np.exp(93.4517 / (temperature_in_kelvin / 100) - 60.2409 + 23.3585 \
            * np.log(temperature_in_kelvin / 100) + salinity * (
                0.023517 - 0.023656 * temperature_in_kelvin / 100 + 0.0047036 * (temperature_in_kelvin / 100) ** 2))

    # Now calculate FugFac according to Weiss (1974) Marine Chemestry TODO: What does FugFac mean?
    # delta_x and b_x are in cm3/mol TODO we should use m over cm right?
    rt_x = 83.1451 * temperature_in_kelvin  # Gas constant times temperature
    delta_x = (57.7 - 0.118 * temperature_in_kelvin)
    b_x = -1636.75 + 12.0408 * temperature_in_kelvin - 0.0327957 * temperature_in_kelvin ** 2 \
            + 3.16528 * 1e-5 * temperature_in_kelvin ** 3
    FugFac = np.exp((b_x + 2 * delta_x) / rt_x)


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

    def ta_iter_SWS(x):
        x2 = x * x
        x3 = x2 * x
        k12 = k1 * k2
        k12p = k1p * k2p
        k123p = k12p * k3p
        c = 1.0 + st / ks + ft / kf
        a = x3 + k1p * x2 + k12p * x + k123p
        # print(np.abs(a.min()))
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

    x1 = 10.0 ** (-ph_high)
    x2 = 10.0 ** (-ph_low)
    xacc = 1e-10

    in1 = np.ones_like(co2_in) * x1
    in2 = np.ones_like(co2_in) * x2
    # hSWS = drtsafe(in1, in2, ta_iter_SWS, xacc)
    hSWS = drtsafe(vs, in1, in2, ta_iter_SWS, xacc)
    vs.hSWS[...] = hSWS


    # Calculate [CO2*] as defined in DOE Methods Handbook 1993 Ver. 2,
    # ORNL/CDIC-74, Dickson and Goyet, eds. (Ch 2 p 1+, Eq A.49)
    co2star = dic * hSWS**2 / (hSWS**2 + k1 * hSWS + k1 * k2)
    co2starair = co2 * ff * atmospheric_pressure
    dco2star = co2starair - co2star
    ph = -np.log10(hSWS)
    # print(np.min(dco2star), np.max(dco2star))

    pCO2 = co2star / (k0 * FugFac)
    dpCO2 = pCO2 - co2starair


    # Convert units back
    co2star /= permil
    dco2star /= permil

    pCO2 /= permeg
    dpCO2 /= permeg

    # return co2star, dco2star
    # print()
    # print("hSWS=", np.min(hSWS), np.min(np.abs(hSWS)), np.max(hSWS))
    # print("co2star=", np.min(co2star), np.min(np.abs(co2star)), np.max(co2star))
    # print("co2starair=", np.min(co2starair), np.min(np.abs(co2starair)), np.max(co2starair))
    # print("dco2star=", np.min(dco2star), np.min(np.abs(dco2star)), np.max(dco2star))
    # print()
    return dco2star



# def drtsafe(guess_low, guess_high, ta_iter_SWS, accuracy, max_iterations=100):
def drtsafe(vs, guess_low, guess_high, ta_iter_SWS, accuracy, max_iterations=100):
    f_low, _ = ta_iter_SWS(guess_low)
    f_high, _ = ta_iter_SWS(guess_high)


    # Switch variables depending on value of f_low
    x_low, x_high, f_low, f_high = np.where(f_low < 0,
            (guess_low, guess_high, f_low, f_high),
            (guess_high, guess_low, f_high, f_low))


    drtsafe_val = 0.5 * (guess_low + guess_high)
    dx = np.abs(guess_high - guess_low)
    # drtsafe_val = vs.hSWS  # Use current value as initial guess
    # dx = np.abs(guess_high - vs.hSWS)
    dx_old = dx.copy()
    f, df = ta_iter_SWS(drtsafe_val)

    mask = np.zeros_like(drtsafe_val, dtype=np.bool)

    for _ in range(max_iterations):
        # print(_, mask.sum())
        tmp_mask = ((drtsafe_val - x_high) * df - f) * ((drtsafe_val - x_low) * df - f) >= 0
        tmp_mask = np.logical_or(tmp_mask, (np.abs(2.0 * f) > np.abs(dx_old * df)))

        dx_old = dx.copy()
        # dx, drtsafe_val = np.where(tmp_mask,
        #         (0.5 * (x_high - x_low), x_low + dx),
        #         (f / df, drtsafe_val - dx))
        dx = np.where(tmp_mask, 0.5 * (x_high - x_low), f/df)
        drtsafe_val = np.where(tmp_mask, x_low + dx, drtsafe_val - dx)

        mask = np.abs(dx) < accuracy

        if mask.all():
            break

        f, df = ta_iter_SWS(drtsafe_val)

        x_low, x_high, f_low, f_high = np.where(f < 0,
                (drtsafe_val, x_high, f, f_high),
                (x_low, drtsafe_val, f_low, f))

        # if not mask.any():
        #     break

        # print("f = ", np.min(f), np.min(np.abs(f)), np.max(f))
        # print("df = ", np.min(df), np.min(np.abs(df)), np.max(df))

    return drtsafe_val
