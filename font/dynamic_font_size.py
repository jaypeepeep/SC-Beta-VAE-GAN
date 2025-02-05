from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

def get_font_sizes():
    app = QApplication.instance()
    if not app:
        app = QApplication([])
    
    screen = app.primaryScreen()
    screen_size = screen.size()
    width = screen_size.width()
    height = screen_size.height()
    
    # Base font sizes for a 1920x1080 resolution
    base_width = 1920
    base_height = 1080
    
    scale_factor_w = width / base_width
    scale_factor_h = height / base_height
    scale_factor = min(scale_factor_w, scale_factor_h)  # Maintain aspect ratio
    
    font_sizes = {
        "button": int(14 * scale_factor),
        "content": int(18 * scale_factor),
        "title": int(24 * scale_factor),
        "subtitle": int(20 * scale_factor),
        "subcontent": int(12 * scale_factor),
        "sidebar": int(18 * scale_factor)
    }
    
    return font_sizes

def apply_fonts(widget, font_family):
    font_sizes = get_font_sizes()
    
    widget.setFont(QFont(font_family, font_sizes["content"]))  # Default font size
    
    # Assuming widget has child elements categorized
    for btn in widget.findChildren(QFont):
        btn.setFont(QFont(font_family, font_sizes["button"]))
    for title in widget.findChildren(QFont):
        title.setFont(QFont(font_family, font_sizes["title"]))
    for subcontent in widget.findChildren(QFont):
        subcontent.setFont(QFont(font_family, font_sizes["subcontent"]))
    for sidebar in widget.findChildren(QFont):
        sidebar.setFont(QFont(font_family, font_sizes["sidebar"]))
