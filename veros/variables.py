from collections import OrderedDict

from . import veros_method, runtime_settings


class Variable:
    def __init__(self, name, dims, units, long_description, dtype=None,
                 output=False, time_dependent=True, scale=1.,
                 write_to_restart=False, extra_attributes=None):
        self.name = name
        self.dims = dims
        self.units = units
        self.long_description = long_description
        self.dtype = dtype
        self.output = output
        self.time_dependent = time_dependent
        self.scale = scale
        self.write_to_restart = write_to_restart
        # : Additional attributes to be written in netCDF output
        self.extra_attributes = extra_attributes or {}


# fill value for netCDF output (invalid data is replaced by this value)
FILL_VALUE = -1e18

#
XT = ('xt',)
XU = ('xu',)
YT = ('yt',)
YU = ('yu',)
ZT = ('zt',)
ZW = ('zw',)
T_HOR = ('xt', 'yt')
U_HOR = ('xu', 'yt')
V_HOR = ('xt', 'yu')
ZETA_HOR = ('xu', 'yu')
T_GRID = ('xt', 'yt', 'zt')
U_GRID = ('xu', 'yt', 'zt')
V_GRID = ('xt', 'yu', 'zt')
W_GRID = ('xt', 'yt', 'zw')
ZETA_GRID = ('xu', 'yu', 'zt')
TIMESTEPS = ('timesteps',)
ISLE = ('isle',)
TENSOR_COMP = ('tensor1', 'tensor2')

# those are written to netCDF output by default
BASE_DIMENSIONS = XT + XU + YT + YU + ZT + ZW + ISLE
GHOST_DIMENSIONS = ('xt', 'yt', 'xu', 'yu')


def get_dimensions(vs, grid, include_ghosts=True, local=True):
    px, py = runtime_settings.num_proc

    dimensions = {
        'xt': vs.nx,
        'xu': vs.nx,
        'yt': vs.ny,
        'yu': vs.ny,
        'zt': vs.nz,
        'zw': vs.nz,
        'timesteps': 3,
        'tensor1': 2,
        'tensor2': 2,
        'isle': vs.nisle,
    }

    if local:
        dimensions.update({
            'xt': dimensions['xt'] // px,
            'xu': dimensions['xu'] // px,
            'yt': dimensions['yt'] // py,
            'yu': dimensions['yt'] // py
        })

    if include_ghosts:
        for d in GHOST_DIMENSIONS:
            dimensions[d] += 4

    dims = []
    for grid_dim in grid:
        if grid_dim in dimensions:
            dims.append(dimensions[grid_dim])
        elif isinstance(grid_dim, int):
            dims.append(grid_dim)
        elif hasattr(vs, grid_dim):
            dims.append(getattr(vs, grid_dim))
        else:
            raise ValueError('unrecognized dimension %s' % grid_dim)

    return tuple(dims)


def remove_ghosts(array, dims):
    ghost_mask = tuple(slice(2, -2) if dim in GHOST_DIMENSIONS else slice(None) for dim in dims)
    return array[ghost_mask]


@veros_method
def add_ghosts(vs, array, dims):
    full_shape = tuple([i + 4 if dim in GHOST_DIMENSIONS else i for i,
                        dim in zip(array.shape, dims)])
    newarr = np.zeros(full_shape, dtype=array.dtype)
    ghost_mask = tuple(slice(2, -2) if dim in GHOST_DIMENSIONS else slice(None) for dim in dims)
    newarr[ghost_mask] = array
    return newarr


def get_grid_mask(vs, grid):
    masks = {
        T_HOR: vs.maskT[:, :, -1],
        U_HOR: vs.maskU[:, :, -1],
        V_HOR: vs.maskV[:, :, -1],
        ZETA_HOR: vs.maskZ[:, :, -1],
        T_GRID: vs.maskT,
        U_GRID: vs.maskU,
        V_GRID: vs.maskV,
        W_GRID: vs.maskW,
        ZETA_GRID: vs.maskZ
    }
    if len(grid) > 2:
        if grid[:3] in masks.keys():
            return masks[grid[:3]]
    if len(grid) > 1:
        if grid[:2] in masks.keys():
            return masks[grid[:2]]
    return None


MAIN_VARIABLES = OrderedDict([
    ('dxt', Variable(
        'Zonal T-grid spacing', XT, 'm',
        'Zonal (x) spacing of T-grid point',
        output=True, time_dependent=False
    )),
    ('dxu', Variable(
        'Zonal U-grid spacing', XU, 'm',
        'Zonal (x) spacing of U-grid point',
        output=True, time_dependent=False
    )),
    ('dyt', Variable(
        'Meridional T-grid spacing', YT, 'm',
        'Meridional (y) spacing of T-grid point',
        output=True, time_dependent=False
    )),
    ('dyu', Variable(
        'Meridional U-grid spacing', YU, 'm',
        'Meridional (y) spacing of U-grid point',
        output=True, time_dependent=False
    )),
    ('zt', Variable(
        'Vertical coordinate (T)', ZT, 'm', 'Vertical coordinate',
        output=True, time_dependent=False, extra_attributes={'positive': 'up'}
    )),
    ('zw', Variable(
        'Vertical coordinate (W)', ZW, 'm', 'Vertical coordinate', output=True,
        time_dependent=False, extra_attributes={'positive': 'up'}
    )),
    ('dzt', Variable(
        'Vertical spacing (T)', ZT, 'm', 'Vertical spacing', output=True, time_dependent=False
    )),
    ('dzw', Variable(
        'Vertical spacing (W)', ZW, 'm', 'Vertical spacing', output=True, time_dependent=False
    )),
    ('cost', Variable(
        'Metric factor (T)', YT, '1', 'Metric factor for spherical coordinates',
        time_dependent=False
    )),
    ('cosu', Variable(
        'Metric factor (U)', YU, '1', 'Metric factor for spherical coordinates',
        time_dependent=False
    )),
    ('tantr', Variable(
        'Metric factor', YT, '1', 'Metric factor for spherical coordinates',
        time_dependent=False
    )),
    ('coriolis_t', Variable(
        'Coriolis frequency', T_HOR, '1/s',
        'Coriolis frequency at T grid point', time_dependent=False
    )),
    ('coriolis_h', Variable(
        'Horizontal Coriolis frequency', T_HOR, '1/s',
        'Horizontal Coriolis frequency at T grid point', time_dependent=False
    )),

    ('kbot', Variable(
        'Index of deepest cell', T_HOR, '',
        'Index of the deepest grid cell (counting from 1, 0 means all land)',
        dtype='int', time_dependent=False
    )),
    ('ht', Variable(
        'Total depth (T)', T_HOR, 'm', 'Total depth of the water column', output=True,
        time_dependent=False
    )),
    ('hu', Variable(
        'Total depth (U)', U_HOR, 'm', 'Total depth of the water column', output=True,
        time_dependent=False
    )),
    ('hv', Variable(
        'Total depth (V)', V_HOR, 'm', 'Total depth of the water column', output=True,
        time_dependent=False
    )),
    ('hur', Variable(
        'Total depth (U), masked', U_HOR, 'm',
        'Total depth of the water column (masked)', time_dependent=False
    )),
    ('hvr', Variable(
        'Total depth (V), masked', V_HOR, 'm',
        'Total depth of the water column (masked)', time_dependent=False
    )),
    ('beta', Variable(
        'Change of Coriolis freq.', T_HOR, '1/(ms)',
        'Change of Coriolis frequency with latitude', output=True, time_dependent=False
    )),
    ('area_t', Variable(
        'Area of T-box', T_HOR, 'm^2', 'Area of T-box', output=True, time_dependent=False
    )),
    ('area_u', Variable(
        'Area of U-box', U_HOR, 'm^2', 'Area of U-box', output=True, time_dependent=False
    )),
    ('area_v', Variable(
        'Area of V-box', V_HOR, 'm^2', 'Area of V-box', output=True, time_dependent=False
    )),

    ('maskT', Variable(
        'Mask for tracer points', T_GRID, '',
        'Mask in physical space for tracer points', dtype='int8', time_dependent=False
    )),
    ('maskU', Variable(
        'Mask for U points', U_GRID, '',
        'Mask in physical space for U points', dtype='int8', time_dependent=False
    )),
    ('maskV', Variable(
        'Mask for V points', V_GRID, '',
        'Mask in physical space for V points', dtype='int8', time_dependent=False
    )),
    ('maskW', Variable(
        'Mask for W points', W_GRID, '',
        'Mask in physical space for W points', dtype='int8', time_dependent=False
    )),
    ('maskZ', Variable(
        'Mask for Zeta points', ZETA_GRID, '',
        'Mask in physical space for Zeta points', dtype='int8', time_dependent=False
    )),

    ('rho', Variable(
        'Density', T_GRID + TIMESTEPS, 'kg/m^3',
        'In-situ density anomaly, relative to the surface mean value of 1024 kg/m^3',
        output=True, write_to_restart=True
    )),

    ('prho', Variable(
        'Potential density', T_GRID, 'kg/m^3',
        'Potential density anomaly, relative to the surface mean value of 1024 kg/m^3 '
        '(equal to in-situ density anomaly for equation of state 1 to 4)',
        output=True
    )),

    ('int_drhodT', Variable(
        'Der. of dyn. enthalpy by temperature', T_GRID + TIMESTEPS, '?',
        'Partial derivative of dynamic enthalpy by temperature', output=True,
        write_to_restart=True
    )),
    ('int_drhodS', Variable(
        'Der. of dyn. enthalpy by salinity', T_GRID + TIMESTEPS, '?',
        'Partial derivative of dynamic enthalpy by salinity', output=True,
        write_to_restart=True
    )),
    ('Nsqr', Variable(
        'Square of stability frequency', W_GRID + TIMESTEPS, '1/s^2',
        'Square of stability frequency', output=True, write_to_restart=True
    )),
    ('Hd', Variable(
        'Dynamic enthalpy', T_GRID + TIMESTEPS, 'm^2/s^2', 'Dynamic enthalpy',
        output=True, write_to_restart=True
    )),
    ('dHd', Variable(
        'Change of dyn. enth. by adv.', T_GRID + TIMESTEPS, 'm^2/s^3',
        'Change of dynamic enthalpy due to advection', write_to_restart=True
    )),

    ('temp', Variable(
        'Temperature', T_GRID + TIMESTEPS, 'deg C',
        'Conservative temperature', output=True, write_to_restart=True
    )),
    ('dtemp', Variable(
        'Temperature tendency', T_GRID + TIMESTEPS, 'deg C/s',
        'Conservative temperature tendency', write_to_restart=True
    )),
    ('salt', Variable(
        'Salinity', T_GRID + TIMESTEPS, 'g/kg', 'Salinity', output=True,
        write_to_restart=True
    )),
    ('dsalt', Variable(
        'Salinity tendency', T_GRID + TIMESTEPS, 'g/(kg s)',
        'Salinity tendency', write_to_restart=True
    )),
    ('dtemp_vmix', Variable(
        'Change of temp. by vertical mixing', T_GRID, 'deg C/s',
        'Change of temperature due to vertical mixing',
    )),
    ('dtemp_hmix', Variable(
        'Change of temp. by horizontal mixing', T_GRID, 'deg C/s',
        'Change of temperature due to horizontal mixing',
    )),
    ('dsalt_vmix', Variable(
        'Change of sal. by vertical mixing', T_GRID, 'deg C/s',
        'Change of salinity due to vertical mixing',
    )),
    ('dsalt_hmix', Variable(
        'Change of sal. by horizontal mixing', T_GRID, 'deg C/s',
        'Change of salinity due to horizontal mixing',
    )),
    ('dtemp_iso', Variable(
        'Change of temp. by isop. mixing', T_GRID, 'deg C/s',
        'Change of temperature due to isopycnal mixing plus skew mixing',
    )),
    ('dsalt_iso', Variable(
        'Change of sal. by isop. mixing', T_GRID, 'deg C/s',
        'Change of salinity due to isopycnal mixing plus skew mixing',

    )),
    ('forc_temp_surface', Variable(
        'Surface temperature flux', T_HOR, 'm K/s', 'Surface temperature flux',
        output=True
    )),
    ('forc_salt_surface', Variable(
        'Surface salinity flux', T_HOR, 'm g/s kg', 'Surface salinity flux',
        output=True
    )),

    ('flux_east', Variable(
        'Multi-purpose flux', U_GRID, '?', 'Multi-purpose flux'
    )),
    ('flux_north', Variable(
        'Multi-purpose flux', V_GRID, '?', 'Multi-purpose flux'
    )),
    ('flux_top', Variable(
        'Multi-purpose flux', W_GRID, '?', 'Multi-purpose flux'
    )),

    ('u', Variable(
        'Zonal velocity', U_GRID + TIMESTEPS, 'm/s', 'Zonal velocity',
        output=True, write_to_restart=True
    )),
    ('v', Variable(
        'Meridional velocity', V_GRID + TIMESTEPS, 'm/s', 'Meridional velocity',
        output=True, write_to_restart=True
    )),
    ('w', Variable(
        'Vertical velocity', W_GRID + TIMESTEPS, 'm/s', 'Vertical velocity',
        output=True, write_to_restart=True
    )),
    ('du', Variable(
        'Zonal velocity tendency', U_GRID + TIMESTEPS, 'm/s',
        'Zonal velocity tendency', write_to_restart=True
    )),
    ('dv', Variable(
        'Meridional velocity tendency', V_GRID + TIMESTEPS, 'm/s',
        'Meridional velocity tendency', write_to_restart=True
    )),
    ('du_cor', Variable(
        'Change of u by Coriolis force', U_GRID, 'm/s^2',
        'Change of u due to Coriolis force'
    )),
    ('dv_cor', Variable(
        'Change of v by Coriolis force', V_GRID, 'm/s^2',
        'Change of v due to Coriolis force'
    )),
    ('du_mix', Variable(
        'Change of u by vertical mixing', U_GRID, 'm/s^2',
        'Change of u due to implicit vertical mixing'
    )),
    ('dv_mix', Variable(
        'Change of v by vertical mixing', V_GRID, 'm/s^2',
        'Change of v due to implicit vertical mixing'
    )),
    ('du_adv', Variable(
        'Change of u by advection', U_GRID, 'm/s^2',
        'Change of u due to advection'
    )),
    ('dv_adv', Variable(
        'Change of v by advection', V_GRID, 'm/s^2',
        'Change of v due to advection'
    )),
    ('p_hydro', Variable(
        'Hydrostatic pressure', T_GRID, 'm^2/s^2', 'Hydrostatic pressure', output=True
    )),
    ('kappaM', Variable(
        'Vertical viscosity', T_GRID, 'm^2/s', 'Vertical viscosity', output=True
    )),
    ('kappaH', Variable(
        'Vertical diffusivity', W_GRID, 'm^2/s', 'Vertical diffusivity', output=True
    )),
    ('surface_taux', Variable(
        'Surface wind stress', U_HOR, 'N/s^2', 'Zonal surface wind stress', output=True,
    )),
    ('surface_tauy', Variable(
        'Surface wind stress', V_HOR, 'N/s^2', 'Meridional surface wind stress', output=True,
    )),
    ('forc_rho_surface', Variable(
        'Surface density flux', T_HOR, '?', 'Surface potential density flux', output=True
    )),

    ('psi', Variable(
        'Streamfunction', ZETA_HOR + TIMESTEPS, 'm^3/s', 'Barotropic streamfunction',
        output=True, write_to_restart=True
    )),
    ('dpsi', Variable(
        'Streamfunction tendency', ZETA_HOR + TIMESTEPS, 'm^3/s^2',
        'Streamfunction tendency', write_to_restart=True
    )),
    ('land_map', Variable(
        'Land map', T_HOR, '', 'Land map'
    )),
    ('isle', Variable(
        'Island number', ISLE, '', 'Island number', output=True
    )),
    ('psin', Variable(
        'Boundary streamfunction', ZETA_HOR + ISLE, 'm^3/s',
        'Boundary streamfunction', output=True, time_dependent=False
    )),
    ('dpsin', Variable(
        'Boundary streamfunction factor', ISLE + TIMESTEPS, '?',
        'Boundary streamfunction factor', write_to_restart=True
    )),
    ('line_psin', Variable(
        'Boundary line integrals', ISLE + ISLE, '?',
        'Boundary line integrals', time_dependent=False
    )),
    ('boundary_mask', Variable(
        'Boundary mask', T_HOR + ISLE, '',
        'Boundary mask', time_dependent=False
    )),
    ('line_dir_south_mask', Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False
    )),
    ('line_dir_north_mask', Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False
    )),
    ('line_dir_east_mask', Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False
    )),
    ('line_dir_west_mask', Variable(
        'Line integral mask', T_HOR + ISLE, '',
        'Line integral mask', time_dependent=False
    )),

    ('K_gm', Variable(
        'Skewness diffusivity', W_GRID, 'm^2/s',
        'GM diffusivity, either constant or from EKE model'
    )),
    ('K_iso', Variable(
        'Isopycnal diffusivity', W_GRID, 'm^2/s', 'Along-isopycnal diffusivity'
    )),

    ('K_diss_v', Variable(
        'Dissipation of kinetic Energy', W_GRID, 'm^2/s^3',
        'Kinetic energy dissipation by vertical, rayleigh and bottom friction',
        write_to_restart=True
    )),
    ('K_diss_bot', Variable(
        'Dissipation of kinetic Energy', W_GRID, 'm^2/s^3',
        'Mean energy dissipation by bottom and rayleigh friction'
    )),
    ('K_diss_h', Variable(
        'Dissipation of kinetic Energy', W_GRID, 'm^2/s^3',
        'Kinetic energy dissipation by horizontal friction'
    )),
    ('K_diss_gm', Variable(
        'Dissipation of mean energy', W_GRID, 'm^2/s^3',
        'Mean energy dissipation by GM (TRM formalism only)'
    )),
    ('P_diss_v', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by vertical diffusion'
    )),
    ('P_diss_nonlin', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by nonlinear equation of state'
    )),
    ('P_diss_iso', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by isopycnal mixing'
    )),
    ('P_diss_skew', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by GM (w/o TRM)'
    )),
    ('P_diss_hmix', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by horizontal mixing'
    )),
    ('P_diss_adv', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by advection'
    )),
    ('P_diss_comp', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by compression'
    )),
    ('P_diss_sources', Variable(
        'Dissipation of potential Energy', W_GRID, 'm^2/s^3',
        'Potential energy dissipation by external sources (e.g. restoring zones)'
    )),

    ('u_wgrid', Variable(
        'U on W grid', W_GRID, 'm/s', 'Zonal velocity interpolated to W grid points'
    )),
    ('v_wgrid', Variable(
        'V on W grid', W_GRID, 'm/s', 'Meridional velocity interpolated to W grid points'
    )),
    ('w_wgrid', Variable(
        'W on W grid', W_GRID, 'm/s', 'Vertical velocity interpolated to W grid points'
    ))
])

CONDITIONAL_VARIABLES = OrderedDict([
    ('coord_degree', OrderedDict([
        ('xt', Variable(
            'Zonal coordinate (T)', XT, 'degrees_east',
            'Zonal (x) coordinate of T-grid point',
            output=True, time_dependent=False
        )),
        ('xu', Variable(
            'Zonal coordinate (U)', XU, 'degrees_east',
            'Zonal (x) coordinate of U-grid point',
            output=True, time_dependent=False
        )),
        ('yt', Variable(
            'Meridional coordinate (T)', YT, 'degrees_north',
            'Meridional (y) coordinate of T-grid point',
            output=True, time_dependent=False
        )),
        ('yu', Variable(
            'Meridional coordinate (U)', YU, 'degrees_north',
            'Meridional (y) coordinate of U-grid point',
            output=True, time_dependent=False
        )),
    ])),

    ('not coord_degree', OrderedDict([
        ('xt', Variable(
            'Zonal coordinate (T)', XT, 'km',
            'Zonal (x) coordinate of T-grid point',
            output=True, scale=1e-3, time_dependent=False
        )),
        ('xu', Variable(
            'Zonal coordinate (U)', XU, 'km',
            'Zonal (x) coordinate of U-grid point',
            output=True, scale=1e-3, time_dependent=False
        )),
        ('yt', Variable(
            'Meridional coordinate (T)', YT, 'km',
            'Meridional (y) coordinate of T-grid point',
            output=True, scale=1e-3, time_dependent=False
        )),
        ('yu', Variable(
            'Meridional coordinate (U)', YU, 'km',
            'Meridional (y) coordinate of U-grid point',
            output=True, scale=1e-3, time_dependent=False
        )),
    ])),

    ('enable_tempsalt_sources', OrderedDict([
        ('temp_source', Variable(
            'Source of temperature', T_GRID, 'K/s',
            'Non-conservative source of temperature', output=True
        )),
        ('salt_source', Variable(
            'Source of salt', T_GRID, 'g/(kg s)',
            'Non-conservative source of salt', output=True
        )),
    ])),

    ('enable_momentum_sources', OrderedDict([
        ('u_source', Variable(
            'Source of zonal velocity', U_GRID, 'm/s^2 (?)',
            'Non-conservative source of zonal velocity', output=True
        )),
        ('v_source', Variable(
            'Source of meridional velocity', V_GRID, 'm/s^2 (?)',
            'Non-conservative source of meridional velocity', output=True
        )),
    ])),

    ('enable_neutral_diffusion', OrderedDict([
        ('K_11', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('K_13', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('K_22', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('K_23', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('K_31', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('K_32', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('K_33', Variable('Isopycnal mixing coefficient', T_GRID, '?', 'Isopycnal mixing tensor component')),
        ('Ai_ez', Variable('?', T_GRID + TENSOR_COMP, '?', '?')),
        ('Ai_nz', Variable('?', T_GRID + TENSOR_COMP, '?', '?')),
        ('Ai_bx', Variable('?', T_GRID + TENSOR_COMP, '?', '?')),
        ('Ai_by', Variable('?', T_GRID + TENSOR_COMP, '?', '?')),
    ])),
    ('enable_skew_diffusion', OrderedDict([
        ('B1_gm', Variable(
            'Zonal component of GM streamfunction', V_GRID, 'm^2/s',
            'Zonal component of GM streamfunction'
        )),
        ('B2_gm', Variable(
            'Meridional component of GM streamfunction', U_GRID, 'm^2/s',
            'Meridional component of GM streamfunction'
        ))
    ])),
    ('enable_bottom_friction_var', OrderedDict([
        ('r_bot_var_u', Variable(
            'Bottom friction coeff.', U_HOR, '?', 'Zonal bottom friction coefficient'
        )),
        ('r_bot_var_v', Variable(
            'Bottom friction coeff.', V_HOR, '?', 'Meridional bottom friction coefficient'
        )),
    ])),
    ('enable_TEM_friction', OrderedDict([
        ('kappa_gm', Variable('Vertical diffusivity', W_GRID, 'm^2/s', 'Vertical diffusivity')),
    ])),
    ('enable_tke', OrderedDict([
        ('tke', Variable(
            'Turbulent kinetic energy', W_GRID + TIMESTEPS, 'm^2/s^2',
            'Turbulent kinetic energy', output=True, write_to_restart=True
        )),
        ('sqrttke', Variable(
            'Square-root of TKE', W_GRID, 'm/s', 'Square-root of TKE'
        )),
        ('dtke', Variable(
            'Turbulent kinetic energy tendency', W_GRID + TIMESTEPS, 'm^2/s^3',
            'Turbulent kinetic energy tendency', write_to_restart=True
        )),
        ('Prandtlnumber', Variable('Prandtl number', W_GRID, '', 'Prandtl number')),
        ('mxl', Variable('Mixing length', W_GRID, 'm', 'Mixing length')),
        ('forc_tke_surface', Variable(
            'TKE surface flux', T_HOR, 'm^3/s^3', 'TKE surface flux', output=True
        )),
        ('tke_diss', Variable(
            'TKE dissipation', W_GRID, 'm^2/s^3', 'TKE dissipation'
        )),
        ('tke_surf_corr', Variable(
            'Correction of TKE surface flux', T_HOR, 'm^3/s^3',
            'Correction of TKE surface flux'
        )),
    ])),
    ('enable_eke', OrderedDict([
        ('eke', Variable(
            'meso-scale energy', W_GRID + TIMESTEPS, 'm^2/s^2',
            'meso-scale energy', output=True, write_to_restart=True
        )),
        ('deke', Variable(
            'meso-scale energy tendency', W_GRID + TIMESTEPS, 'm^2/s^3',
            'meso-scale energy tendency', write_to_restart=True
        )),
        ('sqrteke', Variable(
            'square-root of eke', W_GRID, 'm/s', 'square-root of eke'
        )),
        ('L_rossby', Variable('Rossby radius', T_HOR, 'm', 'Rossby radius')),
        ('L_rhines', Variable('Rhines scale', W_GRID, 'm', 'Rhines scale')),
        ('eke_len', Variable('Eddy length scale', T_GRID, 'm', 'Eddy length scale')),
        ('eke_diss_iw', Variable(
            'Dissipation of EKE to IW', W_GRID, 'm^2/s^3',
            'Dissipation of EKE to internal waves'
        )),
        ('eke_diss_tke', Variable(
            'Dissipation of EKE to TKE', W_GRID, 'm^2/s^3',
            'Dissipation of EKE to TKE'
        )),
        ('eke_bot_flux', Variable(
            'Flux by bottom friction', T_HOR, 'm^3/s^3', 'Flux by bottom friction'
        )),
    ])),
    ('enable_eke_leewave_dissipation', OrderedDict([
        ('eke_topo_hrms', Variable(
            '?', T_HOR, '?', '?'
        )),
        ('eke_topo_lam', Variable(
            '?', T_HOR, '?', '?'
        )),
        ('hrms_k0', Variable(
            '?', T_HOR, '?', '?'
        )),
        ('c_lee', Variable(
            'Lee wave dissipation coefficient', T_HOR, '1/s',
            'Lee wave dissipation coefficient'
        )),
        ('eke_lee_flux', Variable(
            'Lee wave flux', T_HOR, 'm^3/s^3', 'Lee wave flux',
        )),
        ('c_Ri_diss', Variable(
            'Interior dissipation coefficient', W_GRID, '1/s',
            'Interior dissipation coefficient'
        )),
    ])),
    ('enable_idemix', OrderedDict([
        ('E_iw', Variable(
            'Internal wave energy', W_GRID + TIMESTEPS, 'm^2/s^2',
            'Internal wave energy', output=True, write_to_restart=True
        )),
        ('dE_iw', Variable(
            'Internal wave energy tendency', W_GRID + TIMESTEPS, 'm^2/s^2',
            'Internal wave energy tendency', write_to_restart=True
        )),
        ('c0', Variable(
            'Vertical IW group velocity', W_GRID, 'm/s',
            'Vertical internal wave group velocity'
        )),
        ('v0', Variable(
            'Horizontal IW group velocity', W_GRID, 'm/s',
            'Horizontal internal wave group velocity'
        )),
        ('alpha_c', Variable('?', W_GRID, '?', '?')),
        ('iw_diss', Variable(
            'IW dissipation', W_GRID, 'm^2/s^3', 'Internal wave dissipation'
        )),
        ('forc_iw_surface', Variable(
            'IW surface forcing', T_HOR, 'm^3/s^3',
            'Internal wave surface forcing', time_dependent=False, output=True
        )),
        ('forc_iw_bottom', Variable(
            'IW bottom forcing', T_HOR, 'm^3/s^3',
            'Internal wave bottom forcing', time_dependent=False, output=True
        )),
    ])),
])


@veros_method
def get_standard_variables(vs):
    variables = {}

    for var_name, var in MAIN_VARIABLES.items():
        variables[var_name] = var

    for condition, var_dict in CONDITIONAL_VARIABLES.items():
        if condition.startswith('not '):
            eval_condition = not bool(getattr(vs, condition[4:]))
        else:
            eval_condition = bool(getattr(vs, condition))
        if eval_condition:
            for var_name, var in var_dict.items():
                variables[var_name] = var

    return variables


@veros_method(inline=True)
def allocate(vs, dimensions, dtype=None, include_ghosts=True, local=True, fill=0):
    if dtype is None:
        dtype = vs.default_float_type

    shape = get_dimensions(vs, dimensions, include_ghosts=include_ghosts, local=local)
    out = np.empty(shape, dtype=dtype)
    out[...] = fill
    return out
