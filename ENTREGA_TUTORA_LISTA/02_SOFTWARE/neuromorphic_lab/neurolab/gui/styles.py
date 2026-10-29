"""
neurolab.gui.styles
===================
Estilos consistentes para la interfaz gráfica de Neuromorphic Lab (PySide6 / Qt6).
"""

from PySide6.QtGui import QColor, QFont

# --- Palette de Colores en Formato Hexadecimal ---
COLORS = {
    # Fondo
    'bg_dark':      '#f0f0f0',
    'bg_panel':     '#ffffff',
    'bg_canvas':    '#f8f9fa',
    'bg_grid':      '#e9ecef',

    # Elementos
    'sensor':       '#1e3a8a',   # Azul oscuro
    'memristor_nv': '#991b1b',   # Rojo oscuro
    'memristor_v':  '#9a3412',   # Naranja oscuro
    'neuron':       '#166534',   # Verde oscuro
    'actuator':     '#6b21a8',   # Púrpura oscuro
    'wire':         '#9ca3af',   # Gris

    # Estados
    'selected':     '#fef08a',   # Amarillo
    'hover':        '#111827',   # Negro
    'active':       '#0d9488',   # Verde turquesa

    # Texto
    'text':         '#111827',
    'text_dim':     '#4b5563',
    'text_label':   '#374151',

    # Alertas
    'warning':      '#d97706',
    'error':        '#dc2626',
    'success':      '#16a34a',
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
