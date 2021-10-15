



module linear_eq_of_state
!==========================================================================
!  linear equation of state 
!  input is Salinity sa in g/kg, 
!  pot. temperature ct in deg C 
!==========================================================================
 implicit none
 real*8,parameter,private :: rho0 = 1024.0,theta0 = 283.0-273.15, S0 = 35.0
 real*8,parameter,private :: betaT = 1.67d-4, betaS = 0.78d-3 
 real*8,parameter,private :: grav = 9.81, z0=0.0
 contains

 real*8 function linear_eq_of_state_rho(sa,ct)
   real*8,intent(in) :: sa,ct
   linear_eq_of_state_rho = - (betaT*(ct-theta0) -betaS*(sa-S0) )*rho0
 end function

 real*8 function linear_eq_of_state_dyn_enthalpy(sa,ct,p)
  real*8 :: sa,ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  linear_eq_of_state_dyn_enthalpy = grav*zz*(-betaT*thetas+betaS*(sa-S0) ) 
 end function

 real*8 function linear_eq_of_state_salt(rho,ct)
   real*8,intent(in) :: rho,ct
   linear_eq_of_state_salt = (rho +  betaT*(ct-theta0)*rho0 )/(betaS*rho0) + S0
 end function

 real*8 function linear_eq_of_state_drhodT()
   linear_eq_of_state_drhodT = - betaT*rho0
 end function

 real*8 function linear_eq_of_state_drhodS()
   linear_eq_of_state_drhodS =  betaS*rho0
 end function

 real*8 function linear_eq_of_state_drhodp()
   linear_eq_of_state_drhodp =  0.0
 end function
end module linear_eq_of_state




module nonlin1_eq_of_state
!==========================================================================
!  non-linear equation of state from Vallis 2008
!  input is Salinity sa in g/kg, 
!  pot. temperature ct in deg C ,  no pressure dependency
!==========================================================================
 implicit none
 real*8,parameter,private :: rho0 = 1024.0,theta0 = 283.0-273.15, S0 = 35.0
 real*8,parameter,private :: betaT = 1.67d-4, betaTs = 1d-5/2., betaS = 0.78d-3
 real*8,parameter,private :: grav = 9.81, z0=0.0
 contains

 real*8 function nonlin1_eq_of_state_rho(sa,ct)
   real*8 :: sa,ct,thetas
   thetas = ct-theta0
   nonlin1_eq_of_state_rho = - (betaT*thetas + betaTs*thetas**2-betaS*(sa-S0) )*rho0
 end function

 real*8 function nonlin1_eq_of_state_dyn_enthalpy(sa,ct,p)
  real*8 :: sa,ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin1_eq_of_state_dyn_enthalpy = grav*zz*(-betaT*thetas-betaTs*thetas**2+betaS*(sa-S0) ) 
 end function

 real*8 function nonlin1_eq_of_state_salt(rho,ct)
   real*8 :: rho,ct,thetas
   thetas = ct-theta0
   nonlin1_eq_of_state_salt = (rho +  (betaT*thetas + betaTs*thetas**2 )*rho0 )/(betaS*rho0) + S0
 end function

 real*8 function nonlin1_eq_of_state_drhodT(ct)
   real*8 :: ct,thetas
   thetas = ct-theta0
   nonlin1_eq_of_state_drhodT = - (betaT + 2*betaTs*thetas)*rho0
 end function

 real*8 function nonlin1_eq_of_state_drhodS()
   nonlin1_eq_of_state_drhodS =  betaS*rho0
 end function

 real*8 function nonlin1_eq_of_state_drhodp()
   nonlin1_eq_of_state_drhodp =  0.0
 end function
end module nonlin1_eq_of_state



module nonlin2_eq_of_state
!==========================================================================
!  non-linear equation of state from Vallis 2008
!  input is Salinity sa in g/kg, 
!  pot. temperature ct in deg C and 
!  pressure p in dbar
!==========================================================================
 implicit none
 real*8,parameter,private :: rho0 = 1024.0,z0 = 0.0, theta0 = 283.0-273.15, S0 = 35.0
 real*8,parameter,private :: grav=9.81, cs0 = 1490.0, betaT = 1.67d-4, betaTs = 1d-5
 real*8,parameter,private :: betaS = 0.78d-3, gammas = 1.1d-8
 contains

 real*8 function nonlin2_eq_of_state_rho(sa,ct,p)
  real*8 :: sa,ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin2_eq_of_state_rho = &
          - (grav*zz/cs0**2 +betaT*(1-gammas*grav*zz*rho0)*thetas + betaTs/2*thetas**2-betaS*(sa-S0) )*rho0
 end function
 
 real*8 function nonlin2_eq_of_state_dyn_enthalpy(sa,ct,p)
  real*8 :: sa,ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin2_eq_of_state_dyn_enthalpy = grav*0.5*zz**2*( -grav/cs0**2 + betaT*grav*rho0*gammas*thetas ) &
                                    +grav*zz*(-betaT*thetas-betaTs*thetas**2+betaS*(sa-S0) ) 
 end function

 real*8 function nonlin2_eq_of_state_salt(rho,ct,p)
  real*8 :: rho,ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin2_eq_of_state_salt = &
       (rho/rho0 +  (grav*zz/cs0**2 +betaT*(1-gammas*grav*zz*rho0)*thetas + betaTs/2*thetas**2 ))/betaS + S0
 end function
 
 real*8 function nonlin2_eq_of_state_drhodT(ct,p)
  real*8 :: ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin2_eq_of_state_drhodT = - ( betaT*(1-gammas*grav*zz*rho0) + betaTs*thetas )*rho0
 end function
 
 real*8 function nonlin2_eq_of_state_drhodS()
  nonlin2_eq_of_state_drhodS = betaS*rho0
 end function
 
 real*8 function nonlin2_eq_of_state_drhodp(ct)
  real*8 :: ct,thetas
  thetas = ct-theta0
  nonlin2_eq_of_state_drhodp = 1/cs0**2 -betaT*gammas*rho0*thetas  
 end function

 real*8 function nonlin2_eq_of_state_int_drhodT(ct,p)
  real*8 :: ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin2_eq_of_state_int_drhodT = rho0*zz*(betaT+betaTs*thetas) - rho0*betaT*gammas*grav*rho0*zz**2/2 
 end function

 real*8 function nonlin2_eq_of_state_int_drhodS(p)
  real*8 :: p, zz
  zz=-p-z0
  nonlin2_eq_of_state_int_drhodS = -betaS*rho0*zz
 end function

end module nonlin2_eq_of_state






module nonlin3_eq_of_state
!==========================================================================
!  non-linear equation of state, no salinity dependency
!  input is Salinity sa in g/kg, 
!  pot. temperature ct in deg C ,  no pressure dependency
!==========================================================================
 implicit none
 real*8,parameter,private :: rho0 = 1024.0,theta0 = 283.0-273.15, S0 = 35.0
 real*8,parameter,private :: betaT = 1.67d-4, betaTs = 1d-5/2., betaS = 0
 real*8,parameter,private :: grav = 9.81, z0=0.0
 contains

 real*8 function nonlin3_eq_of_state_rho(sa,ct)
   real*8 :: sa,ct,thetas
   thetas = ct-theta0
   nonlin3_eq_of_state_rho = - (betaT*thetas + betaTs*thetas**2-betaS*(sa-S0) )*rho0
 end function

 real*8 function nonlin3_eq_of_state_dyn_enthalpy(sa,ct,p)
  real*8 :: sa,ct,p, zz,thetas
  zz=-p-z0
  thetas = ct-theta0
  nonlin3_eq_of_state_dyn_enthalpy = grav*zz*(-betaT*thetas-betaTs*thetas**2+betaS*(sa-S0) ) 
 end function

 real*8 function nonlin3_eq_of_state_salt(rho,ct)
   real*8 :: rho,ct,thetas
   thetas = ct-theta0
   nonlin3_eq_of_state_salt = (rho +  (betaT*thetas + betaTs*thetas**2 )*rho0 )/(betaS*rho0) + S0
 end function

 real*8 function nonlin3_eq_of_state_drhodT(ct)
   real*8 :: ct,thetas
   thetas = ct-theta0
   nonlin3_eq_of_state_drhodT = - (betaT + 2*betaTs*thetas)*rho0
 end function

 real*8 function nonlin3_eq_of_state_drhodS()
   nonlin3_eq_of_state_drhodS =  betaS*rho0
 end function

 real*8 function nonlin3_eq_of_state_drhodp()
   nonlin3_eq_of_state_drhodp =  0.0
 end function
end module nonlin3_eq_of_state





module gsw_eq_of_state
 !==========================================================================
 !  in-situ density, dynamic enthalpy and derivatives
 !  from Absolute Salinity and Conservative 
 !  Temperature, using the computationally-efficient 48-term expression for
 !  density in terms of SA, CT and p (IOC et al., 2010).
 !==========================================================================
 implicit none

      real*8, private, parameter :: v01 =  9.998420897506056d+2 
      real*8, private, parameter :: v02 =  2.839940833161907d0
      real*8, private, parameter :: v03 = -3.147759265588511d-2 
      real*8, private, parameter :: v04 =  1.181805545074306d-3
      real*8, private, parameter :: v05 = -6.698001071123802d0 
      real*8, private, parameter :: v06 = -2.986498947203215d-2
      real*8, private, parameter :: v07 =  2.327859407479162d-4 
      real*8, private, parameter :: v08 = -3.988822378968490d-2
      real*8, private, parameter :: v09 =  5.095422573880500d-4 
      real*8, private, parameter :: v10 = -1.426984671633621d-5
      real*8, private, parameter :: v11 =  1.645039373682922d-7 
      real*8, private, parameter :: v12 = -2.233269627352527d-2
      real*8, private, parameter :: v13 = -3.436090079851880d-4 
      real*8, private, parameter :: v14 =  3.726050720345733d-6
      real*8, private, parameter :: v15 = -1.806789763745328d-4 
      real*8, private, parameter :: v16 =  6.876837219536232d-7
      real*8, private, parameter :: v17 = -3.087032500374211d-7 
      real*8, private, parameter :: v18 = -1.988366587925593d-8
      real*8, private, parameter :: v19 = -1.061519070296458d-11 
      real*8, private, parameter :: v20 =  1.550932729220080d-10
      real*8, private, parameter :: v21 =  1.0d0
      real*8, private, parameter :: v22 =  2.775927747785646d-3 
      real*8, private, parameter :: v23 = -2.349607444135925d-5
      real*8, private, parameter :: v24 =  1.119513357486743d-6 
      real*8, private, parameter :: v25 =  6.743689325042773d-10
      real*8, private, parameter :: v26 = -7.521448093615448d-3 
      real*8, private, parameter :: v27 = -2.764306979894411d-5
      real*8, private, parameter :: v28 =  1.262937315098546d-7 
      real*8, private, parameter :: v29 =  9.527875081696435d-10
      real*8, private, parameter :: v30 = -1.811147201949891d-11 
      real*8, private, parameter :: v31 = -3.303308871386421d-5
      real*8, private, parameter :: v32 =  3.801564588876298d-7 
      real*8, private, parameter :: v33 = -7.672876869259043d-9
      real*8, private, parameter :: v34 = -4.634182341116144d-11 
      real*8, private, parameter :: v35 =  2.681097235569143d-12
      real*8, private, parameter :: v36 =  5.419326551148740d-6 
      real*8, private, parameter :: v37 = -2.742185394906099d-5
      real*8, private, parameter :: v38 = -3.212746477974189d-7 
      real*8, private, parameter :: v39 =  3.191413910561627d-9
      real*8, private, parameter :: v40 = -1.931012931541776d-12 
      real*8, private, parameter :: v41 = -1.105097577149576d-7
      real*8, private, parameter :: v42 =  6.211426728363857d-10 
      real*8, private, parameter :: v43 = -1.119011592875110d-10
      real*8, private, parameter :: v44 = -1.941660213148725d-11 
      real*8, private, parameter :: v45 = -1.864826425365600d-14
      real*8, private, parameter :: v46 =  1.119522344879478d-14 
      real*8, private, parameter :: v47 = -1.200507748551599d-15
      real*8, private, parameter :: v48 =  6.057902487546866d-17 
      real*8, parameter, private :: rho0 = 1024.0

  contains

  !==========================================================================
  real*8 function gsw_rho(sa,ct,p) 
  ! density as a function of T, S, and p
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
       real*8, intent(in)  :: sa, ct, p
       real*8  :: sqrtsa, v_hat_denominator, v_hat_numerator
       sqrtsa = sqrt(sa)
       v_hat_denominator = v01 + ct*(v02 + ct*(v03 + v04*ct))  &
              + sa*(v05 + ct*(v06 + v07*ct) &
          + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct)))) &
               + p*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct) &
               + p*(v17 + ct*(v18 + v19*ct) + v20*sa))
       v_hat_numerator = v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct)))  &
            + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa  &
        + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct)))))   &
             + p*(v37 + ct*(v38 + ct*(v39 + v40*ct))   &
            + sa*(v41 + v42*ct) + p*(v43 + ct*(v44 + v45*ct + v46*sa)  &
             + p*(v47 + v48*ct))) 
       gsw_rho = v_hat_denominator/v_hat_numerator - rho0
      end function


  !==========================================================================
  real*8 function gsw_drhodT(sa, ct, p)
  ! d/dT of density
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
       real*8, intent(in) :: sa, ct, p
       real*8 , parameter :: a01 =  2.839940833161907d0 
       real*8, parameter :: a02 = -6.295518531177023d-2
       real*8 ,parameter :: a03 =  3.545416635222918d-3 
       real*8, parameter :: a04 = -2.986498947203215d-2
       real*8 ,parameter :: a05 =  4.655718814958324d-4 
       real*8, parameter :: a06 =  5.095422573880500d-4
       real*8 ,parameter :: a07 = -2.853969343267241d-5 
       real*8, parameter :: a08 =  4.935118121048767d-7
       real*8 ,parameter :: a09 = -3.436090079851880d-4 
       real*8, parameter :: a10 =  7.452101440691467d-6
       real*8 ,parameter :: a11 =  6.876837219536232d-7 
       real*8, parameter :: a12 = -1.988366587925593d-8
       real*8 ,parameter :: a13 = -2.123038140592916d-11 
       real*8, parameter :: a14 =  2.775927747785646d-3
       real*8 ,parameter :: a15 = -4.699214888271850d-5 
       real*8, parameter :: a16 =  3.358540072460230d-6
       real*8 ,parameter :: a17 =  2.697475730017109d-9 
       real*8, parameter :: a18 = -2.764306979894411d-5
       real*8 ,parameter :: a19 =  2.525874630197091d-7 
       real*8, parameter :: a20 =  2.858362524508931d-9
       real*8 ,parameter :: a21 = -7.244588807799565d-11 
       real*8, parameter :: a22 =  3.801564588876298d-7
       real*8 ,parameter :: a23 = -1.534575373851809d-8 
       real*8, parameter :: a24 = -1.390254702334843d-10
       real*8 ,parameter :: a25 =  1.072438894227657d-11 
       real*8, parameter :: a26 = -3.212746477974189d-7
       real*8 ,parameter :: a27 =  6.382827821123254d-9 
       real*8, parameter :: a28 = -5.793038794625329d-12
       real*8 ,parameter :: a29 =  6.211426728363857d-10 
       real*8, parameter :: a30 = -1.941660213148725d-11
       real*8 ,parameter :: a31 = -3.729652850731201d-14 
       real*8, parameter :: a32 =  1.119522344879478d-14
       real*8 ,parameter :: a33 =  6.057902487546866d-17
       real*8 ::  sqrtsa, v_hat_denominator, v_hat_numerator
       real*8 :: dvhatden_dct, dvhatnum_dct, rho, rec_num

       sqrtsa = sqrt(sa)
       v_hat_denominator = v01 + ct*(v02 + ct*(v03 + v04*ct))+ sa*(v05 + ct*(v06 + v07*ct)  &
              + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct)))) + p*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct)  &
              + p*(v17 + ct*(v18 + v19*ct) + v20*sa))

       v_hat_numerator = v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct))) &
            + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa &
            + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct))))) + p*(v37 + ct*(v38 + ct*(v39 + v40*ct))  &
            + sa*(v41 + v42*ct) + p*(v43 + ct*(v44 + v45*ct + v46*sa) + p*(v47 + v48*ct)))
       
       dvhatden_dct = a01 + ct*(a02 + a03*ct) + sa*(a04 + a05*ct + sqrtsa*(a06 + ct*(a07 + a08*ct)))  &
         + p*(a09 + a10*ct + a11*sa + p*(a12 + a13*ct))

       dvhatnum_dct = a14 + ct*(a15 + ct*(a16 + a17*ct)) + sa*(a18 + ct*(a19 + ct*(a20 + a21*ct)) &
         + sqrtsa*(a22 + ct*(a23 + ct*(a24 + a25*ct)))) &
         + p*(a26 + ct*(a27 + a28*ct) + a29*sa + p*(a30 + a31*ct + a32*sa + a33*p))

       rec_num = 1d0/v_hat_numerator
       rho = rec_num*v_hat_denominator
       gsw_drhodT=(dvhatden_dct-dvhatnum_dct*rho)*rec_num
      end function

  !==========================================================================
  real*8 function gsw_drhodS(sa, ct, p)
  ! d/dS of density
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
        real*8, intent(in) :: sa, ct, p
        real*8 , parameter :: b01 = -6.698001071123802d0 
        real*8 , parameter :: b02 = -2.986498947203215d-2
        real*8 , parameter :: b03 =  2.327859407479162d-4 
        real*8 , parameter :: b04 = -5.983233568452735d-2
        real*8 , parameter :: b05 =  7.643133860820750d-4 
        real*8 , parameter :: b06 = -2.140477007450431d-5
        real*8 , parameter :: b07 =  2.467559060524383d-7 
        real*8 , parameter :: b08 = -1.806789763745328d-4
        real*8 , parameter :: b09 =  6.876837219536232d-7 
        real*8 , parameter :: b10 =  1.550932729220080d-10
        real*8 , parameter :: b11 = -7.521448093615448d-3 
        real*8 , parameter :: b12 = -2.764306979894411d-5
        real*8 , parameter :: b13 =  1.262937315098546d-7 
        real*8 , parameter :: b14 =  9.527875081696435d-10
        real*8 , parameter :: b15 = -1.811147201949891d-11 
        real*8 , parameter :: b16 = -4.954963307079632d-5
        real*8 , parameter :: b17 =  5.702346883314446d-7 
        real*8 , parameter :: b18 = -1.150931530388857d-8
        real*8 , parameter :: b19 = -6.951273511674217d-11 
        real*8 , parameter :: b20 =  4.021645853353715d-12
        real*8 , parameter :: b21 =  1.083865310229748d-5 
        real*8 , parameter :: b22 = -1.105097577149576d-7
        real*8 , parameter :: b23 =  6.211426728363857d-10 
        real*8 , parameter :: b24 =  1.119522344879478d-14
        real*8 :: sqrtsa, v_hat_denominator, v_hat_numerator
        real*8 :: dvhatden_dsa, dvhatnum_dsa, rho, rec_num

        sqrtsa = sqrt(sa)
        v_hat_denominator = v01 + ct*(v02 + ct*(v03 + v04*ct)) + sa*(v05 + ct*(v06 + v07*ct)  &
              + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct)))) + p*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct)  &
              + p*(v17 + ct*(v18 + v19*ct) + v20*sa))

        v_hat_numerator = v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct)))  &
           + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa  &
            + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct)))))   &
            + p*(v37 + ct*(v38 + ct*(v39 + v40*ct)) + sa*(v41 + v42*ct) + p*(v43 + ct*(v44 + v45*ct + v46*sa)  &
            + p*(v47 + v48*ct)))
       
        dvhatden_dsa = b01 + ct*(b02 + b03*ct) + sqrtsa*(b04 + ct*(b05 + ct*(b06 + b07*ct))) &
          + p*(b08 + b09*ct + b10*p) 

        dvhatnum_dsa = b11 + ct*(b12 + ct*(b13 + ct*(b14 + b15*ct))) &
          + sqrtsa*(b16 + ct*(b17 + ct*(b18 + ct*(b19 + b20*ct)))) &
          + b21*sa + p*(b22 + ct*(b23 + b24*p))

        rec_num = 1d0/v_hat_numerator
        rho = rec_num*v_hat_denominator
        gsw_drhodS= (dvhatden_dsa-dvhatnum_dsa*rho)*rec_num
      end function


  !==========================================================================
  real*8 function gsw_drhodp(sa, ct, p)
  ! d/dp of density
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
        real*8, intent(in) :: sa, ct, p
        real*8 , parameter :: c01 = -2.233269627352527d-2  
        real*8 , parameter :: c02 = -3.436090079851880d-4
        real*8 , parameter :: c03 =  3.726050720345733d-6  
        real*8 , parameter :: c04 = -1.806789763745328d-4
        real*8 , parameter :: c05 =  6.876837219536232d-7  
        real*8 , parameter :: c06 = -6.174065000748422d-7
        real*8 , parameter :: c07 = -3.976733175851186d-8  
        real*8 , parameter :: c08 = -2.123038140592916d-11
        real*8 , parameter :: c09 =  3.101865458440160d-10 
        real*8 , parameter :: c10 = -2.742185394906099d-5
        real*8 , parameter :: c11 = -3.212746477974189d-7  
        real*8 , parameter :: c12 =  3.191413910561627d-9
        real*8 , parameter :: c13 = -1.931012931541776d-12 
        real*8 , parameter :: c14 = -1.105097577149576d-7
        real*8 , parameter :: c15 =  6.211426728363857d-10 
        real*8 , parameter :: c16 = -2.238023185750219d-10
        real*8 , parameter :: c17 = -3.883320426297450d-11 
        real*8 , parameter :: c18 = -3.729652850731201d-14
        real*8 , parameter :: c19 =  2.239044689758956d-14 
        real*8 , parameter :: c20 = -3.601523245654798d-15
        real*8 , parameter :: c21 =  1.817370746264060d-16, pa2db = 1d-4
        real*8 :: sqrtsa, v_hat_denominator, v_hat_numerator
        real*8 :: dvhatden_dp, dvhatnum_dp, rho, rec_num

        sqrtsa = sqrt(sa)
        v_hat_denominator = v01 + ct*(v02 + ct*(v03 + v04*ct)) + sa*(v05 + ct*(v06 + v07*ct)  &
              + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct)))) + p*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct)  &
              + p*(v17 + ct*(v18 + v19*ct) + v20*sa))

        v_hat_numerator = v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct)))  &
           + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa &
            + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct)))))   &
            + p*(v37 + ct*(v38 + ct*(v39 + v40*ct)) + sa*(v41 + v42*ct) + p*(v43 + ct*(v44 + v45*ct + v46*sa)  &
            + p*(v47 + v48*ct)))

        dvhatden_dp = c01 + ct*(c02 + c03*ct) + sa*(c04 + c05*ct) + p*(c06 + ct*(c07 + c08*ct) + c09*sa)

        dvhatnum_dp = c10 + ct*(c11 + ct*(c12 + c13*ct)) &
         + sa*(c14 + c15*ct) + p*(c16 + ct*(c17 + c18*ct + c19*sa) + p*(c20 + c21*ct))

        rec_num = 1d0/v_hat_numerator
        rho = rec_num*v_hat_denominator
        gsw_drhodp =  pa2db*(dvhatden_dp-dvhatnum_dp*rho)*rec_num
      end function


  !==========================================================================
  real*8 function gsw_dyn_enthalpy(sa,ct,p) 
  !==========================================================================

  ! Calculates dynamic enthalpy of seawater using the computationally
  ! efficient 48-term expression for density in terms of SA, CT and p
  ! (IOC et al., 2010)
  ! 
  ! A component due to the constant reference density in Boussinesq 
  ! approximation is removed
  !
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]

      implicit none
       real*8, intent(in) :: sa, ct, p
      real*8 :: db2pa, sqrtsa, a0, a1, a2, a3, b0, b1, b2
      real*8 :: sqrt_disc, ca, cb, cn, cm, part, b1sq, Hd

      ! Hd = - g/rho_0 int_z^0 rho dz' + g z = - g/rho_0 int_z^0 rho dz' - p_0 /rho_0
      ! - g/rho_0 int_z^0 rho dz'  = Hd + p_0 /rho_0

      db2pa = 1d4                             ! factor to convert from dbar to Pa
      sqrtsa = sqrt(sa)
      a0 = v21 + ct*(v22 + ct*(v23 + ct*(v24 + v25*ct)))  &
         + sa*(v26 + ct*(v27 + ct*(v28 + ct*(v29 + v30*ct))) + v36*sa   &
         + sqrtsa*(v31 + ct*(v32 + ct*(v33 + ct*(v34 + v35*ct)))))
      a1 = v37 + ct*(v38 + ct*(v39 + v40*ct)) + sa*(v41 + v42*ct)
      a2 = v43 + ct*(v44 + v45*ct + v46*sa)
      a3 = v47 + v48*ct
      b0 = v01 + ct*(v02 + ct*(v03 + v04*ct)) + sa*(v05 + ct*(v06 + v07*ct)  &
            + sqrtsa*(v08 + ct*(v09 + ct*(v10 + v11*ct))))
      b1 = 0.5d0*(v12 + ct*(v13 + v14*ct) + sa*(v15 + v16*ct))
      b2 = v17 + ct*(v18 + v19*ct) + v20*sa
      b1sq = b1*b1 
      sqrt_disc = sqrt(b1sq - b0*b2)
      cn = a0 + (2*a3*b0*b1/b2 - a2*b0)/b2
      cm = a1 + (4*a3*b1sq/b2 - a3*b0 - 2*a2*b1)/b2
      ca = b1 - sqrt_disc
      cb = b1 + sqrt_disc
      part = (cn*b2 - cm*b1)/(b2*(cb - ca))
      Hd = db2pa*(p*(a2 - 2d0*a3*b1/b2 + 0.5d0*a3*p)/b2 + (cm/(2d0*b2))*log(1d0 + p*(2d0*b1 + b2*p)/b0)   &
                  + part*log(1d0 + (b2*p*(cb - ca))/(ca*(cb + b2*p))))
      gsw_dyn_enthalpy =  Hd- p*db2pa/rho0
  end function

  !==========================================================================
  real*8 function gsw_dHdT1(sa,ct,p)
  ! d/dT of dynamic enthalpy, numerical derivative
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
      implicit none
      real*8, intent(in) :: sa, ct, p
      real*8,parameter :: delta = 1d-4
      gsw_dHdT1 = (gsw_dyn_enthalpy(sa,ct+delta,p)-gsw_dyn_enthalpy(sa,ct,p))/delta
  end function


  !==========================================================================
  real*8 function gsw_dHdS1(sa,ct,p)
  ! d/dS of dynamic enthalpy, numerical derivative
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
      implicit none
      real*8, intent(in) :: sa, ct, p
      real*8,parameter :: delta = 1d-4
      gsw_dHdS1 = (gsw_dyn_enthalpy(sa+delta,ct,p)-gsw_dyn_enthalpy(sa,ct,p))/delta
  end function


  !==========================================================================
  real*8 function gsw_dHdT(sa_in,ct_in,p)
  ! d/dT of dynamic enthalpy, analytical derivative
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
      implicit real (kind=8) (a-h,o-z)
      real*8, intent(in) :: sa_in, ct_in, p
      real*8 :: sa,ct
      sa = max(1d-1,sa_in) ! prevent division by zero
      ct = max(-12d0,ct_in)  ! prevent blowing up for values smaller than -15 degC
      t1 = v45 * ct
      t2 = 0.2D1 * t1
      t3 = v46 * sa
      t4 = 0.5D0 * v12
      t5 = v14 * ct
      t7 = ct * (v13 + t5)
      t8 = 0.5D0 * t7
      t11 = sa * (v15 + v16 * ct)
      t12 = 0.5D0 * t11
      t13 = t4 + t8 + t12
      t15 = v19 * ct
      t19 = v17 + ct * (v18 + t15) + v20 * sa
      t20 = 0.1D1 / t19
      t24 = v47 + v48 * ct
      t25 = 0.5D0 * v13
      t26 = 0.10D1 * t5
      t27 = sa * v16
      t28 = 0.5D0 * t27
      t29 = t25 + t26 + t28
      t33 = t24 * t13
      t34 = t19 ** 2
      t35 = 0.1D1 / t34
      t37 = v18 + 0.2D1 * t15
      t38 = t35 * t37
      t48 = ct * (v44 + t1 + t3)
      t57 = v40 * ct
      t59 = ct * (v39 + t57)
      t64 = t13 ** 2
      t68 = t20 * t29
      t71 = t24 * t64
      t74 = v04 * ct
      t76 = ct * (v03 + t74)
      t79 = v07 * ct
      t82 = sqrt(sa)
      t83 = v11 * ct
      t85 = ct * (v10 + t83)
      t92 = v01 + ct * (v02 + t76) + sa * (v05 + ct * (v06 + t79) + t82 * (v08 + ct * (v09 + t85)))
      t93 = v48 * t92
      t105 = v02 + t76 + ct * (v03+0.2D1*t74)+sa*(v06 + 0.2D1*t79 + t82 * (v09 + t85 + ct * (v10 + 0.2D1 * t83)))
      t106 = t24 * t105
      t107 = v44 + t2 + t3
      t110 = v43 + t48
      t117 = t24 * t92
      t120 = 0.4D1 * t71 * t20 - t117 - 0.2D1 * t110 * t13
      t123 = v38 + t59 + ct * (v39 + 0.2D1 * t57) + sa * v42 + (0.4D1 * &
          v48 * t64 * t20 + 0.8D1 * t33 * t68 - 0.4D1 * t71 * t38 - t93 - t106 &
         - 0.2D1 * t107 * t13 - 0.2D1 * t110 * t29) * t20 - t120 * t35 * t37
      t128 = t19 * p
      t130 = p * (0.10D1 * v12 + 0.10D1 * t7 + 0.10D1 * t11 + t128)
      t131 = 0.1D1 / t92
      t133 = 0.1D1 + t130 * t131
      t134 = log(t133)
      t143 = v37 + ct * (v38 + t59) + sa * (v41 + v42 * ct) + t120 * t20
      t152 = t37 * p
      t156 = t92 ** 2
      t165 = v25 * ct
      t167 = ct * (v24 + t165)
      t169 = ct * (v23 + t167)
      t175 = v30 * ct
      t177 = ct * (v29 + t175)
      t179 = ct * (v28 + t177)
      t185 = v35 * ct
      t187 = ct * (v34 + t185)
      t189 = ct * (v33 + t187)
      t199 = t13 * t20
      t217 = 0.2D1 * t117 * t199 - t110 * t92
      t234 = v21 + ct * (v22 + t169) + sa * (v26 + ct * (v27 + t179)+v36*sa+t82*(v31+ct*(v32 + t189))) + t217 * t20
      t241 = t64 - t92 * t19
      t242 = sqrt(t241)
      t243 = 0.1D1 / t242
      t244 = t4 + t8 + t12 - t242
      t245 = 0.1D1 / t244
      t247 = t4 + t8 + t12 + t242 + t128
      t248 = 0.1D1 / t247
      t249 = t242 * t245 * t248
      t252 = 0.1D1 + 0.2D1 * t128 * t249
      t253 = log(t252)
      t254 = t243 * t253
      t259 = t234 * t19 - t143 * t13
      t264 = t259 * t20
      t272 = 0.2D1 * t13 * t29 - t105 * t19 - t92 * t37
      t282 = t128 * t242
      t283 = t244 ** 2
      t287 = t243 * t272 / 0.2D1
      t292 = t247 ** 2
      t305 = 0.1D5 * p * (v44 + t2 + t3 - 0.2D1 * v48 * t13 * t20 &
            - 0.2D1*t24 * t29 * t20 + 0.2D1*t33*t38+0.5D0*v48*p)*t20  &
            -0.1D5 * p * (v43 + t48 - 0.2D1 * t33 * t20 + 0.5D0 * t24 * p) * t38 &
            + 0.5D4 * t123 * t20 * t134 - 0.50D4 * t143 * t35 * t134 * t37 &
            + 0.5D4 * t143 * t20 * (p*(0.10D1*v13 + 0.20D1*t5 + 0.10D1 * t27 + t152) * t131 &
            - t130 / t156 * t105) / t133 &
            + 0.5D4 * ((v22 + t169 + ct * (v23 + t167 + ct * (v24 + 0.2D1 * t165)) &
            + sa * (v27 + t179 + ct * (v28 + t177 + ct * (v29 + 0.2D1 * t175)) + t82 * (v32 + t189 &
            + ct * (v33 + t187 + ct*(v34 + 0.2D1 * t185))))+(0.2D1*t93*t199 + 0.2D1*t106*&
       t199 + 0.2D1 * t117 * t68 - 0.2D1 * t117 * t13 * t35 * t37 - t107&
       * t92 - t110 * t105) * t20 - t217 * t35 * t37) * t19 + t234 * t37&
       - t123 * t13 - t143 * t29) * t20 * t254 - 0.5D4 * t259 *&
       t35 * t254 * t37 - 0.25D4 * t264 / t242 / t241 * t253 * t272 &
        + 0.5D4 * t264 * t243 * (0.2D1 * t152 * t249 + t128 *&
       t243 * t245 * t248 * t272 - 0.2D1 * t282 / t283 * t248 * (t25 + t26 + t28 - t287) &
      - 0.2D1 * t282 * t245 / t292 * (t25 + t26 + t28 + t287 + t152)) / t252
      gsw_dHdT = t305
  end function


  !==========================================================================
  real*8 function gsw_dHdS(sa_in,ct_in,p)
  ! d/dS of dynamic enthalpy, analytical derivative
  ! sa     : Absolute Salinity                               [g/kg]
  ! ct     : Conservative Temperature                        [deg C]
  ! p      : sea pressure                                    [dbar]
  !==========================================================================
      implicit real (kind=8) (a-h,o-z)
      real*8, intent(in) :: sa_in, ct_in, p
      real*8 :: sa,ct
      sa = max(1d-1,sa_in) ! prevent division by zero
      ct = max(-12d0, ct_in)  ! prevent blowing up for values smaller than -15 degC
      t1 = ct * v46
      t3 = v47 + v48 * ct
      t4 = 0.5D0 * v15
      t5 = v16 * ct
      t6 = 0.5D0 * t5
      t7 = t4 + t6
      t13 = v17 + ct * (v18 + v19 * ct) + v20 * sa
      t14 = 0.1D1 / t13
      t17 = 0.5D0 * v12
      t20 = ct * (v13 + v14 * ct)
      t21 = 0.5D0 * t20
      t23 = sa * (v15 + t5)
      t24 = 0.5D0 * t23
      t25 = t17 + t21 + t24
      t26 = t3 * t25
      t27 = t13 ** 2
      t28 = 0.1D1 / t27
      t29 = t28 * v20
      t39 = ct * (v44 + v45 * ct + v46 * sa)
      t48 = v42 * ct
      t49 = t14 * t7
      t52 = t25 ** 2
      t53 = t3 * t52
      t58 = ct * (v06 + v07 * ct)
      t59 = sqrt(sa)
      t66 = t59 * (v08 + ct * (v09 + ct * (v10 + v11 * ct)))
      t68 = v05 + t58 + 0.3D1 / 0.2D1 * t66
      t69 = t3 * t68
      t72 = v43 + t39
      t86 = v01 + ct * (v02 + ct * (v03 + v04 * ct)) + sa * (v05 + t58 +t66)
      t87 = t3 * t86
      t90 = 0.4D1 * t53 * t14 - t87 - 0.2D1 * t72 * t25
      t93 = v41 + t48 + (0.8D1 * t26 * t49 - 0.4D1 * t53 * t29 - t69 - 0.2D1*t1 * t25-0.2D1*t72 * t7)*t14 - t90*t28*v20
      t98 = t13 * p
      t100 = p * (0.10D1 * v12 + 0.10D1 * t20 + 0.10D1 * t23 + t98)
      t101 = 0.1D1 / t86
      t103 = 0.1D1 + t100 * t101
      t104 = log(t103)
      t115 = v37 + ct * (v38 + ct * (v39 + v40 * ct)) + sa * (v41 + t48) + t90 * t14
      t123 = v20 * p
      t127 = t86 ** 2
      t142 = ct * (v27 + ct * (v28 + ct * (v29 + v30 * ct)))
      t143 = v36 * sa
      t151 = v31 + ct * (v32 + ct * (v33 + ct * (v34 + v35 * ct)))
      t152 = t59 * t151
      t158 = t25 * t14
      t174 = 0.2D1 * t87 * t158 - t72 * t86
      t189 = v21 + ct * (v22 + ct * (v23 + ct * (v24 + v25 * ct))) + sa*(v26+t142 + t143 + t152)+t174 * t14
      t196 = t52 - t86 * t13
      t197 = sqrt(t196)
      t198 = 0.1D1 / t197
      t199 = t17 + t21 + t24 - t197
      t200 = 0.1D1 / t199
      t202 = t17 + t21 + t24 + t197 + t98
      t203 = 0.1D1 / t202
      t204 = t197 * t200 * t203
      t207 = 0.1D1 + 0.2D1 * t98 * t204
      t208 = log(t207)
      t209 = t198 * t208
      t214 = t189 * t13 - t115 * t25
      t219 = t214 * t14
      t227 = 0.2D1 * t25 * t7 - t68 * t13 - t86 * v20
      t237 = t98 * t197
      t238 = t199 ** 2
      t242 = t198 * t227 / 0.2D1
      t247 = t202 ** 2
      t260 = 0.1D5 * p * (t1 - 0.2D1 * t3 * t7 * t14 + 0.2D1 * t26 * t29) * t14 &
         - 0.1D5 * p * (v43 + t39 - 0.2D1 * t26 * t14 + 0.5D0 * t3* p) * t29 &
         + 0.5D4 * t93 * t14 * t104 - 0.5D4 *t115 * t28 * t104 * v20 &
         + 0.5D4 * t115 * t14 * (p * (0.10D1 * v15 + 0.10D1 * t5 + t123) * t101 - t100 / t127 * t68) / t103 &
         + 0.50D4 * ((v26 + t142 + t143 + t152 + sa * (v36 + 0.1D1/ t59 * t151 / 0.2D1) &
         + (0.2D1 * t69 * t158 + 0.2D1 * t87 * t49 - 0.2D1 * t87 * t25 * t28 * v20 - t1 * t86 - t72 * t68) * t14 &
         - t174 * t28 * v20) * t13 + t189 * v20 - t93 * t25 - t115 * t7) * t14 *t209 - 0.5D4 * t214 * t28 * t209 * v20 &
         - 0.25D4 * t219 / t197 / t196 * t208 * t227 + 0.5D4 * t219 * t198 * (0.2D1 * t123 * t204 &
         + t98 * t198 * t200 * t203 * t227 - 0.2D1 *t237 / t238 * t203 * (t4 + t6 - t242) &
         - 0.2D1 * t237 * t200 / t247 * (t4 + t6 + t242 + t123)) / t207
      gsw_dHdS = t260
  end function

end module gsw_eq_of_state


