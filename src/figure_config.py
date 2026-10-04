import matplotlib.pyplot as plt

def set_paper_style():
    """Apply the figure style used for the paper."""
    plt.rcParams.update({
        # Font sizes
        'font.size': 16,          # Base font size
        'axes.labelsize': 16,     # Axis labels (x, y)
        'axes.titlesize': 16,     # Title
        'xtick.labelsize': 14,    # Tick labels (x)
        'ytick.labelsize': 14,    # Tick labels (y)
        'legend.fontsize': 12,    # Legend
        
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'stix', 
        'axes.formatter.use_mathtext': True,

        # Lines and markers
        'lines.linewidth': 2,
        'lines.markersize': 4,

        # Axes
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # Saving
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 450,
    })