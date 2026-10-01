import matplotlib.pyplot as plt

def set_paper_style():
    """論文用の図のスタイルを一括設定"""
    plt.rcParams.update({
        # フォントサイズ設定
        'font.size': 16,          # 基本のフォントサイズ
        'axes.labelsize': 16,     # 軸ラベル（x, y）
        'axes.titlesize': 16,     # タイトル
        'xtick.labelsize': 14,    # 目盛り（x）
        'ytick.labelsize': 14,    # 目盛り（y）
        'legend.fontsize': 12,    # 凡例
        
        # フォント設定
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'stix', 
        'axes.formatter.use_mathtext': True,

        # 線やマーカーの設定
        'lines.linewidth': 2,
        'lines.markersize': 4,

        # 軸の設定
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # 保存設定
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 450,
    })