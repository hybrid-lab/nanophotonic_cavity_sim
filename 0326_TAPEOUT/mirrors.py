import numpy as np
try:
    import gdsfactory as gf
    gf.gpdk.PDK.activate()
except ImportError:
    gf = None

ROUNDING = .001       # Nearest 1 nm
TOLERANCE = int(-np.log10(ROUNDING))
ROUNDING = 10 ** -TOLERANCE

context = {
    "wavelength" : .78,
    "thickness" : .15,
    "width" : .8
}

mirror_labels = {
    "width" : "Width $w$ [µm]",
    "period" : "Period $\\Lambda$ [µm]",
    "hx" : "Hole Size $x$ [µm]",
    "hy" : "Hole Size $y$ [µm]",
}

def construct_mirrors(
    perturbation_type="period_fillfactor",
    mode_type="air",
    taper_min_feature=.2,
    context=context,
):
    result = {}

    for case in ["defect", "mirror", "taper"]:
        result[case] = {
            "width" : .8,
            "period" : .5 - .1 * (case != "mirror"),
            "hx" : .3,
            "hy" : .4,
        }

        if case == "taper":
            result[case]["hx"] = taper_min_feature
            result[case]["hy"] = taper_min_feature

    return result

def get_mirror_name(parameters):
    name = "mirror"
    for key in mirror_labels:
        if key in parameters:
            # Make mirror parameters even
            value = np.round(parameters[key] / ROUNDING / 2) * ROUNDING * 2
            if isinstance(value, (int, float)):
                value = f"{value}".rstrip("0").rstrip(".")

            name += f"__{key}={value}"

    return name

def error_check_mirror_parameters(parameters):
    if (parameters["hx"] <= 0) or (parameters["hy"] <= 0):
        raise ValueError("Hole sizes must be positive.")
    if parameters["hx"] > parameters["period"]:
        raise ValueError("Hole size hx cannot exceed mirror period.")
    if parameters["hy"] > parameters["width"]:
        raise ValueError("Hole size hy cannot exceed mirror width.")

MIRROR_CACHE = {}

def clear_cache():
    MIRROR_CACHE.clear()
    gf.clear_cache()

def render_mirror(parameters):
    name = get_mirror_name(parameters)

    if not name in MIRROR_CACHE:
        # Error check parameters
        error_check_mirror_parameters(parameters)

        if not "width_left" in parameters:
            parameters["width_left"] = parameters["width"]
        if not "width_right" in parameters:
            parameters["width_right"] = parameters["width"]

        # Actually draw the mirror
        N_circ = 50

        theta = np.linspace(0, 2*np.pi, N_circ, endpoint=True)
        hole_x = (parameters["hx"] / 2) * np.sin(theta)
        hole_y = (parameters["hy"] / 2) * np.cos(theta)

        sy0 = parameters["width_left"] / 2
        sy1 = parameters["width"] / 2
        sy2 = parameters["width_right"] / 2
        sx = parameters["period"] / 2

        square_x = np.array([0, sx, sx, 0, -sx, -sx, 0])
        square_y = np.array([sy1, sy2, -sy2, -sy1, -sy0, sy0, sy1])

        mirror_x = np.concatenate([-square_x, hole_x])
        mirror_y = np.concatenate([square_y, hole_y])

        # Fillout the cached result
        MIRROR_CACHE[name] = parameters.copy()
        polygon = np.column_stack((mirror_x, mirror_y))
        MIRROR_CACHE[name]["polygon"] = polygon

        # Create GDSFactory cell
        if gf is not None:
            mirror_cell = gf.Component(name)
            mirror_cell.add_polygon(polygon, layer=1)

            MIRROR_CACHE[name]["cell"] = mirror_cell

    return MIRROR_CACHE[name]