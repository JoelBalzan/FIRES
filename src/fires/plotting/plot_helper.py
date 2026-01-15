import matplotlib.transforms as mtransforms

from fires.utils.params import param_info

#	--------------------------	Colour maps	---------------------------
#colour blind friendly: https://gist.github.com/thriveth/8560036
colours = {
	'red'   : '#e41a1c',
	'blue'  : '#377eb8',
	'purple': '#984ea3',
	'orange': '#ff7f00',
	'green' : '#4daf4a',
	'pink'  : '#f781bf',
	'brown' : '#a65628',
	'gray'  : '#999999',
	'yellow': '#dede00'
} 

colour_map = {
	'lowest-quarter, total'   : '#e41a1c',
	'highest-quarter, total'  : '#377eb8',
	'full-band, total'        : '#984ea3',
	'full-band, leading'      : '#ff7f00',
	'full-band, trailing'     : '#4daf4a',
	'lower-mid-quarter, total': '#a65628',
	'upper-mid-quarter, total': '#999999',
}

#	--------------------------	Parameter mappings	---------------------------
param_map = {
	# Intrinsic parameters - format: (LaTeX_symbol, unit)
	"tau"         : (r"\tau_0", r"\mathrm{ms}"),
	"width"          : (r"W_0", r"\mathrm{ms}"),
	"A"              : (r"A_0", r"\mathrm{Jy}"),
	"spec_idx"       : (r"\alpha_0", ""),
	"DM"             : (r"\mathrm{DM}_0", r"\mathrm{pc\,cm^{-3}}"),
	"RM"             : (r"\mathrm{RM}_0", r"\mathrm{rad\,m^{-2}}"),
	"PA"             : (r"\psi_0", r"\mathrm{deg}"),
	"lfrac"          : (r"\Pi_{L,0}", ""),
	"vfrac"          : (r"\Pi_{V,0}", ""),
	"dPA"            : (r"\Delta\psi_0", r"\mathrm{deg}"),
	"band_centre_mhz": (r"\nu_{\mathrm{c},0}", r"\mathrm{MHz}"),
	"band_width_mhz" : (r"\Delta \nu_0", r"\mathrm{MHz}"),
	"N"              : (r"N", ""),
	"mg_width_low"   : (r"w_{\mathrm{low},0}", r"\%"),
	"mg_width_high"  : (r"w_{\mathrm{high},0}", r"\%"),
	# sd_<param> 
	"sd_t0"             : (r"\sigma_{t_0}", r"\mathrm{ms}"),
	"sd_A"              : (r"\sigma_A", ""),
	"sd_spec_idx"       : (r"\sigma_\alpha", ""),
	"sd_DM"             : (r"\sigma_{\mathrm{DM}}", r"\mathrm{pc\,cm^{-3}}"),
	"sd_RM"             : (r"\sigma_{\mathrm{RM}}", r"\mathrm{rad\,m^{-2}}"),
	"sd_PA"             : (r"\sigma_{\psi}", r"\mathrm{deg}"),
	"sd_lfrac"          : (r"\sigma_{\Pi_L}", ""),
	"sd_vfrac"          : (r"\sigma_{\Pi_V}", ""),
	"sd_dPA"            : (r"\sigma_{\Delta\psi}", r"\mathrm{deg}"),
	"sd_band_centre_mhz": (r"\sigma_{\nu_c}", r"\mathrm{MHz}"),
	"sd_band_width_mhz" : (r"\sigma_{\Delta \nu}", r"\mathrm{MHz}"),
}


def param_info_or_dynamic(name: str) -> tuple[str, str]:
	"""
	Get (symbol, unit) for a parameter key.
	- First, try param_map (explicit overrides).
	- Otherwise, build from canonical rules in fires.utils.params.
	"""
	if name in param_map:
		val = param_map[name]
		return val if isinstance(val, tuple) else (val, "")
	return param_info(name)


def build_plot_text_string(plot_text, gdict):
	if not plot_text:
		return None
	label_parts = []
	for item in plot_text:
		try:
			if gdict and item in gdict and len(gdict[item]) > 0:
				val = gdict[item][0]
				sym, unit = param_info_or_dynamic(item)
				val_str = str(int(val)) if float(val).is_integer() else f"{float(val):g}"
				label = rf"{sym} = {val_str}" + (rf"~[{unit}]" if unit else "")
				label_parts.append(label)
			else:
				label_parts.append(str(item))
		except Exception:
			label_parts.append(str(item))
	if not label_parts:
		return None
	return r",\; ".join(label_parts)


def text_with_offset(ax, x, y, s, dx_pts=0, dy_pts=0, ha='left', va='bottom',
					  transform='data', color=None, fontsize=None, alpha=None,
					  bbox=None, zorder=None, rotation=None):
	"""
	Draw text at (x, y) with an offset in points (dx_pts, dy_pts).
	- transform='data' or 'axes' selects the base transform.
	"""
	base = ax.transData if transform == 'data' else ax.transAxes
	offset = mtransforms.ScaledTranslation(dx_pts/72.0, dy_pts/72.0, ax.figure.dpi_scale_trans)
	tr = base + offset
	return ax.text(x, y, s, transform=tr, ha=ha, va=va, color=color, fontsize=fontsize,
				   alpha=alpha, bbox=bbox, zorder=zorder, rotation=rotation)


def get_plot_param(plot_config, section, key, default=None):
	"""Helper to safely get plotting parameters"""
	if plot_config is None:
		return default
	sec = plot_config.get(section)
	if key is None:
		return sec if sec is not None else default
	if isinstance(sec, dict):
		return sec.get(key, default)
	return default

				   
def draw_plot_text(ax, display_text, plot_type, plot_config=None):
	if not display_text:
		return
	if plot_type=='general':
		group = 'text_style'
	else:
		group = 'param_text_style'
	style = get_plot_param(plot_config, plot_type, group, {}) or {}
	if not style.get('enabled', True):
		return

	position = style.get('position', 'bottom-right')
	offset_pts = style.get('offset_pts', [0, 0]) or [0, 0]
	colour = style.get('colour', style.get('color', 'gray'))
	alpha = float(style.get('alpha', 1.0))
	fontsize = style.get('fontsize', None)
	zorder = style.get('zorder', 5)

	# Map named positions to axes coordinates
	pos_name = None
	if isinstance(position, (list, tuple)) and len(position) == 2:
		x_pos, y_pos = float(position[0]), float(position[1])
	else:
		pos_name = str(position).strip().lower()
		if pos_name in ('top-left', 'tl'):
			x_pos, y_pos = 0.02, 0.98
		elif pos_name in ('top-right', 'tr'):
			x_pos, y_pos = 0.98, 0.98
		elif pos_name in ('bottom-left', 'bl'):
			x_pos, y_pos = 0.02, 0.01
		else:  # 'bottom-right' or fallback
			x_pos, y_pos = 0.98, 0.01

	ha_cfg = style.get('ha', None)
	va_cfg = style.get('va', None)
	if ha_cfg is not None:
		ha = ha_cfg
	elif pos_name and 'left' in pos_name:
		ha = 'left'
	else:
		ha = 'right'
	if va_cfg is not None:
		va = va_cfg
	elif pos_name and 'top' in pos_name:
		va = 'top'
	else:
		va = 'bottom'

	dx, dy = float(offset_pts[0]), float(offset_pts[1])
	text_with_offset(
		ax, x_pos, y_pos, f"${display_text}$",
		dx_pts=dx, dy_pts=dy,
		ha=ha, va=va,
		transform='axes',
		color=colour, fontsize=fontsize, alpha=alpha,
		zorder=zorder,
	)