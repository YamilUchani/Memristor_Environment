"""
neurolab.gui.styles
===================
Estilos consistentes para la interfaz gráfica de Neuromorphic Lab (PySide6 / Qt6).
"""

from PySide6.QtGui import QColor, QFont

# --- Palette de Colores en Formato Hexadecimal ---
COLORS = {
    # Fondo
    'bg_dark':      '#1e1e2e',
    'bg_panel':     '#252536',
    'bg_canvas':    '#181825',
    'bg_grid':      '#313244',

    # Elementos
    'sensor':       '#89b4fa',   # Azul Catppuccin
    'memristor_nv': '#f38ba8',   # Rojo/Rosa (no volátil)
    'memristor_v':  '#fab387',   # Naranja (volátil)
    'neuron':       '#a6e3a1',   # Verde Catppuccin
    'actuator':     '#cba6f7',   # Púrpura
    'wire':         '#6c7086',   # Gris

    # Estados
    'selected':     '#f9e2af',   # Amarillo
    'hover':        '#ffffff',   # Blanco
    'active':       '#94e2d5',   # Verde turquesa brillante

    # Texto
    'text':         '#cdd6f4',
    'text_dim':     '#a6adc8',
    'text_label':   '#bac2de',

    # Alertas
    'warning':      '#fab387',
    'error':        '#f38ba8',
    'success':      '#a6e3a1',
}


def get_qcolor(name: str) -> QColor:
    """Devuelve un objeto QColor a partir del nombre de la paleta COLORS."""
    hex_code = COLORS.get(name, '#ffffff')
    return QColor(hex_code)


# --- Fuentes PySide6 ---
FONTS = {
    'title':    QFont('Segoe UI', 13, QFont.Weight.Bold),
    'subtitle': QFont('Segoe UI', 11, QFont.Weight.Bold),
    'label':    QFont('Segoe UI', 10),
    'value':    QFont('Consolas', 10),
    'small':    QFont('Segoe UI', 9),
    'mono':     QFont('Consolas', 9),
}

# --- Dimensiones del canvas ---
CANVAS = {
    'width':  900,
    'height': 600,
    'padding': 60,
    'wire_width': 3,
    'element_size': 80,
}
