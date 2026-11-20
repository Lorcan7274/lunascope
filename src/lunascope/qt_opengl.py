"""Shared OpenGL format, configured before QApplication creates any contexts."""
from PySide6.QtGui import QSurfaceFormat


def surface_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    return fmt


def configure_opengl():
    # macOS requires this before QApplication for core-profile context sharing,
    # including the internal context used to compose QOpenGLWidget into a window.
    QSurfaceFormat.setDefaultFormat(surface_format())
