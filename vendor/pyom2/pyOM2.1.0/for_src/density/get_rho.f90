



real*8 function get_rho(salt_loc,temp_loc,press) 
!-----------------------------------------------------------------------
! calculate density as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press

if (eq_of_state_type == 1) then
 get_rho = linear_eq_of_state_rho(salt_loc,temp_loc)
else if (eq_of_state_type == 2) then
 get_rho = nonlin1_eq_of_state_rho(salt_loc,temp_loc)
else if (eq_of_state_type == 3) then
 get_rho = nonlin2_eq_of_state_rho(salt_loc,temp_loc,press)
else if (eq_of_state_type == 4) then
 get_rho = nonlin3_eq_of_state_rho(salt_loc,temp_loc)
else if (eq_of_state_type == 5) then
  get_rho = gsw_rho(salt_loc,temp_loc,press)
else 
 get_rho=0
 call halt_stop(' unknown equation of state in get_rho')
endif
end function get_rho



real*8 function get_dyn_enthalpy(salt_loc,temp_loc,press) 
!-----------------------------------------------------------------------
! calculate dynamic enthalpy as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press

if (eq_of_state_type == 1) then
 get_dyn_enthalpy = linear_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
else if (eq_of_state_type == 2) then
 get_dyn_enthalpy = nonlin1_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
else if (eq_of_state_type == 3) then
 get_dyn_enthalpy = nonlin2_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
else if (eq_of_state_type == 4) then
 get_dyn_enthalpy = nonlin3_eq_of_state_dyn_enthalpy(salt_loc,temp_loc,press)
else if (eq_of_state_type == 5) then
 get_dyn_enthalpy = gsw_dyn_enthalpy(salt_loc,temp_loc,press)
else 
 get_dyn_enthalpy=0
 call halt_stop(' unknown equation of state in get_dyn_enthalpy')
endif
end function get_dyn_enthalpy



real*8 function get_salt(rho_loc,temp_loc,press_loc) 
!-----------------------------------------------------------------------
! calculate salinity as a function of density, temperature and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
implicit none
real*8, intent(in) :: rho_loc,temp_loc,press_loc

if (eq_of_state_type == 1) then
 get_salt = linear_eq_of_state_salt(rho_loc,temp_loc)
else if (eq_of_state_type == 2) then
 get_salt = nonlin1_eq_of_state_salt(rho_loc,temp_loc)
else if (eq_of_state_type == 3) then
 get_salt = nonlin2_eq_of_state_salt(rho_loc,temp_loc,press_loc)
else if (eq_of_state_type == 4) then
 get_salt = nonlin3_eq_of_state_salt(rho_loc,temp_loc)
else 
 get_salt=0
 call halt_stop(' unknown equation of state in get_salt')
endif
end function get_salt


real*8 function get_drhodT(salt_loc,temp_loc,press_loc) 
!-----------------------------------------------------------------------
! calculate drho/dT as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press_loc

if (eq_of_state_type == 1) then
 get_drhodT = linear_eq_of_state_drhodT()
else if (eq_of_state_type == 2) then
 get_drhodT = nonlin1_eq_of_state_drhodT(temp_loc)
else if (eq_of_state_type == 3) then
 get_drhodT = nonlin2_eq_of_state_drhodT(temp_loc,press_loc)
else if (eq_of_state_type == 4) then
 get_drhodT = nonlin3_eq_of_state_drhodT(temp_loc)
else if (eq_of_state_type == 5) then
 get_drhodT = gsw_drhodT(salt_loc,temp_loc,press_loc)
else 
 get_drhodT = 0
 call halt_stop(' unknown equation of state in get_rho')
endif
end function get_drhodT


real*8 function get_drhodS(salt_loc,temp_loc,press_loc) 
!-----------------------------------------------------------------------
! calculate drho/dS as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press_loc

if (eq_of_state_type == 1) then
 get_drhodS = linear_eq_of_state_drhodS()
else if (eq_of_state_type == 2) then
 get_drhodS = nonlin1_eq_of_state_drhodS()
else if (eq_of_state_type == 3) then
 get_drhodS = nonlin2_eq_of_state_drhodS()
else if (eq_of_state_type == 4) then
 get_drhodS = nonlin3_eq_of_state_drhodS()
else if (eq_of_state_type == 5) then
 get_drhodS = gsw_drhodS(salt_loc,temp_loc,press_loc)
else 
 get_drhodS = 0
 call halt_stop(' unknown equation of state in get_rho')
endif
end function get_drhodS


real*8 function get_drhodp(salt_loc,temp_loc,press_loc) 
!-----------------------------------------------------------------------
! calculate drho/dP as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press_loc

if (eq_of_state_type == 1) then
 get_drhodp = linear_eq_of_state_drhodp()
else if (eq_of_state_type == 2) then
 get_drhodp = nonlin1_eq_of_state_drhodp()
else if (eq_of_state_type == 3) then
 get_drhodp = nonlin2_eq_of_state_drhodp(temp_loc)
else if (eq_of_state_type == 4) then
 get_drhodp = nonlin3_eq_of_state_drhodp()
else if (eq_of_state_type == 5) then
 get_drhodp = gsw_drhodp(salt_loc,temp_loc,press_loc)
else 
 get_drhodp = 0
 call halt_stop(' unknown equation of state in get_drhodp')
endif
end function get_drhodP



real*8 function get_int_drhodT(salt_loc,temp_loc,press_loc) 
!-----------------------------------------------------------------------
! calculate int_z^0 drho/dT dz' as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press_loc

if (eq_of_state_type == 1) then
 get_int_drhodT = press_loc*linear_eq_of_state_drhodT()  !  int_z^0rho_T dz = - rho_T z
elseif (eq_of_state_type == 2) then
 get_int_drhodT = press_loc*nonlin1_eq_of_state_drhodT(temp_loc)
elseif (eq_of_state_type == 3) then
 get_int_drhodT = nonlin2_eq_of_state_int_drhodT(temp_loc,press_loc)
elseif (eq_of_state_type == 4) then
 get_int_drhodT = press_loc*nonlin3_eq_of_state_drhodT(temp_loc)
elseif (eq_of_state_type == 5) then
 get_int_drhodT = -(1024.0/9.81)*gsw_dHdT(salt_loc,temp_loc,press_loc)
else 
 get_int_drhodT = 0
 call halt_stop(' unknown equation of state in get_int_drhodT')
endif
end function 


real*8 function get_int_drhodS(salt_loc,temp_loc,press_loc) 
!-----------------------------------------------------------------------
! calculate int_z^0 drho/dS dz' as a function of temperature, salinity and pressure
!-----------------------------------------------------------------------
use main_module
use linear_eq_of_state
use nonlin1_eq_of_state
use nonlin2_eq_of_state
use nonlin3_eq_of_state
use gsw_eq_of_state
implicit none
real*8, intent(in) :: salt_loc,temp_loc,press_loc

if (eq_of_state_type == 1) then
 get_int_drhodS = press_loc*linear_eq_of_state_drhodS()  !  int_z^0rho_T dz = - rho_T z
elseif (eq_of_state_type == 2) then
 get_int_drhodS = press_loc*nonlin1_eq_of_state_drhodS()
elseif (eq_of_state_type == 3) then
 get_int_drhodS = nonlin2_eq_of_state_int_drhodS(press_loc)
elseif (eq_of_state_type == 4) then
 get_int_drhodS = press_loc*nonlin3_eq_of_state_drhodS()
elseif (eq_of_state_type == 5) then
 get_int_drhodS = -(1024.0/9.81)*gsw_dHdS(salt_loc,temp_loc,press_loc)
else 
 get_int_drhodS = 0
 call halt_stop(' unknown equation of state in get_int_rho')
endif
end function 
