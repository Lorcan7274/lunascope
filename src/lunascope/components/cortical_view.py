"""Qt-only cortical renderer. Geometry is static; playback uploads colors only."""
from __future__ import annotations

import time

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import (QColor, QLinearGradient, QMatrix4x4, QPainter,
                           QVector3D)
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram,
                              QOpenGLVertexArrayObject)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .atlas_model import pick_triangle
from ..qt_opengl import surface_format


def parcel_colors(values, face_parcels, cmap, limits):
    values = np.asarray(values, dtype=float)
    lo, hi = limits
    colors = np.full((len(face_parcels), 4), (.32, .34, .37, 1.), dtype=np.float32)
    matched = face_parcels >= 0
    valid = matched.copy()
    valid[matched] = np.isfinite(values[face_parcels[matched]])
    colors[valid] = colormaps[cmap](np.clip((values[face_parcels[valid]] - lo) / (hi - lo), 0, 1))
    return colors


class CorticalView(QOpenGLWidget):
    identified = Signal(int, bool)  # parcel vector index, click vs hover; -1 is neutral
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFormat(surface_format())
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.atlas = None
        self.values = np.array([])
        self.cmap = "viridis"
        self.limits = (0., 1.)
        self.units = ""
        self.hemisphere = "Both"
        self.surface = "pial"
        self._program = None
        self._geometry_dirty = True
        self._colors_dirty = True
        self._buffers = {}
        self._last_hover = 0.
        self._press = None
        self._dragged = False
        self.reset()

    def set_atlas(self, atlas):
        self.atlas = atlas
        self.face_parcels = atlas.face_parcels()
        h = atlas.hemisphere[atlas.faces]
        self.face_hemi = np.where(np.all(h == 0, axis=1), 0,
                                  np.where(np.all(h == 1, axis=1), 1, -1))
        self.values = np.full(len(atlas.parcels), np.nan)
        self.surface = "pial"
        self._geometry_dirty = self._colors_dirty = True
        self.reset()

    def set_values(self, values, cmap, limits, units=""):
        values = np.asarray(values, dtype=float)
        if self.atlas is None or values.shape != (len(self.atlas.parcels),):
            raise ValueError("Expected one value per atlas parcel in table order")
        if not np.isfinite(limits).all() or limits[1] <= limits[0]:
            raise ValueError("Color limits must be finite and increasing")
        self.values, self.cmap, self.limits, self.units = values, cmap, limits, units
        self._colors_dirty = True
        self.update()

    def set_display(self, hemisphere, surface):
        self.hemisphere, self.surface = hemisphere, surface
        self._colors_dirty = True
        self.update()

    def reset(self):
        self.yaw, self.pitch, self.zoom = 0., 90., 1.
        self.update()

    def preset(self, name):
        side = -1 if self.hemisphere == "Right" else 1
        self.yaw, self.pitch = {"Lateral": (side * 90., 0.),
                                "Medial": (-side * 90., 0.),
                                "Superior": (0., 90.), "Inferior": (0., -90.)}[name]
        self.update()

    def initializeGL(self):
        self._gl = self.context().functions()
        self._gl.initializeOpenGLFunctions()
        self.context().aboutToBeDestroyed.connect(self._cleanup)
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        program = QOpenGLShaderProgram(self)
        vertex = """#version 150
            in vec3 position; in vec3 normal; in vec4 color;
            uniform mat4 mvp; uniform mat4 rotation;
            out vec4 shade;
            void main() {
                vec3 n = normalize(mat3(rotation) * normal);
                float light = 0.60 + 0.40 * abs(dot(n, normalize(vec3(.2,.4,1.))));
                shade = vec4(color.rgb * light, color.a);
                gl_Position = mvp * vec4(position, 1.);
            }"""
        fragment = """#version 150
            in vec4 shade; out vec4 frag;
            void main() { if (shade.a < .5) discard; frag = shade; }"""
        if not (program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex)
                and program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment)
                and program.link()):
            self.failed.emit("OpenGL shader initialization failed: " + program.log())
            return
        self._program = program
        self._geometry_dirty = self._colors_dirty = True

    def _cleanup(self):
        self.makeCurrent()
        for buffer in self._buffers.values():
            buffer.destroy()
        self._buffers.clear()
        if hasattr(self, "_vao"):
            self._vao.destroy()
        if self._program is not None:
            self._program.removeAllShaders()
        self._program = None
        self.doneCurrent()

    def _upload(self, name, data, dynamic=False):
        buffer = self._buffers.get(name)
        if buffer is None:
            buffer = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
            buffer.create()
            buffer.setUsagePattern(QOpenGLBuffer.DynamicDraw if dynamic else QOpenGLBuffer.StaticDraw)
            self._buffers[name] = buffer
        buffer.bind()
        payload = np.ascontiguousarray(data, dtype=np.float32).tobytes()
        if dynamic and buffer.size() == len(payload):
            buffer.write(0, payload, len(payload))
        else:
            buffer.allocate(payload, len(payload))
        buffer.release()

    def _visible(self):
        if self.hemisphere == "Both":
            return np.ones(len(self.atlas.faces), dtype=bool)
        return self.face_hemi == (0 if self.hemisphere == "Left" else 1)

    def _matrices(self):
        projection = QMatrix4x4()
        aspect = max(1, self.width()) / max(1, self.height())
        projection.perspective(35., aspect, .1, 30.)
        view = QMatrix4x4()
        view.translate(0., 0., -4.5 / self.zoom)
        rotation = QMatrix4x4()
        # RAS: make superior screen-up and anterior screen-right at left lateral.
        rotation.rotate(self.pitch, 1., 0., 0.)
        rotation.rotate(self.yaw, 0., 1., 0.)
        basis = QMatrix4x4(1., 0., 0., 0., 0., 0., 1., 0.,
                           0., -1., 0., 0., 0., 0., 0., 1.)
        rotation = rotation * basis
        return projection * view * rotation, rotation

    def paintGL(self):
        # QPainter maintains its own GL state. Native GL must be bracketed so
        # Qt restores its shader/VAO/framebuffer state for the overlay/compositor.
        painter = QPainter(self)
        painter.beginNativePainting()
        try:
            self._paint_surface()
        finally:
            painter.endNativePainting()
        self._paint_overlay(painter)
        painter.end()

    def _paint_surface(self):
        gl = self._gl
        gl.glBindFramebuffer(0x8D40, self.defaultFramebufferObject())  # GL_FRAMEBUFFER
        ratio = self.devicePixelRatioF()
        gl.glViewport(0, 0, round(self.width() * ratio), round(self.height() * ratio))
        gl.glDisable(0x0C11)  # GL_SCISSOR_TEST
        gl.glDisable(0x0BE2)  # GL_BLEND
        gl.glDisable(0x0B44)  # GL_CULL_FACE
        gl.glDepthMask(True)
        gl.glColorMask(True, True, True, True)
        gl.glClearColor(.051, .067, .09, 1.)
        gl.glClear(0x4000 | 0x0100)
        if self.atlas is not None and self._program is not None:
            gl.glEnable(0x0B71)  # GL_DEPTH_TEST
            self._vao.bind()
            if self._geometry_dirty:
                self._triangles = {}
                # Normalize every surface using its own fixed bounding sphere.
                for key, vertices in self.atlas.surfaces.items():
                    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
                    scale = np.linalg.norm(vertices - center, axis=1).max()
                    triangles = ((vertices - center) / max(scale, 1e-6))[self.atlas.faces]
                    self._triangles[key] = triangles
                    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
                    self._upload(key + "position", triangles.reshape(-1, 3))
                    self._upload(key + "normal", np.repeat(normals, 3, axis=0))
                self._geometry_dirty = False
            if self._colors_dirty:
                colors = parcel_colors(self.values, self.face_parcels, self.cmap, self.limits)
                colors[~self._visible(), 3] = 0
                self._upload("color", np.repeat(colors, 3, axis=0), True)
                self._colors_dirty = False
            program = self._program
            program.bind()
            mvp, rotation = self._matrices()
            program.setUniformValue("mvp", mvp)
            program.setUniformValue("rotation", rotation)
            for name, size in (("position", 3), ("normal", 3), ("color", 4)):
                self._buffers["color" if name == "color" else self.surface + name].bind()
                location = program.attributeLocation(name)
                program.enableAttributeArray(location)
                program.setAttributeBuffer(location, 0x1406, 0, size, 0)  # GL_FLOAT
            gl.glDrawArrays(0x0004, 0, len(self.atlas.faces) * 3)  # GL_TRIANGLES
            for name in ("position", "normal", "color"):
                program.disableAttributeArray(program.attributeLocation(name))
            program.release()
            self._buffers["color"].release()
            self._vao.release()
            gl.glDisable(0x0B71)

    def _paint_overlay(self, painter):
        painter.setPen(QColor("#c9d1d9"))
        if self.atlas is None:
            painter.drawText(self.rect(), Qt.AlignCenter, "Load an atlas directory to view cortical regions")
        else:
            gradient = QLinearGradient(20, 0, max(21, self.width() - 20), 0)
            for x in np.linspace(0, 1, 32):
                gradient.setColorAt(float(x), QColor.fromRgbF(*map(float, colormaps[self.cmap](x))))
            painter.fillRect(20, self.height() - 42, max(1, self.width() - 40), 12, gradient)
            painter.drawText(20, self.height() - 12, f"{self.limits[0]:.4g}")
            painter.drawText(self.rect().adjusted(20, 0, -20, -5), Qt.AlignRight | Qt.AlignBottom,
                             f"{self.limits[1]:.4g} {self.units}")
            painter.drawText(12, 20, f"{self.hemisphere} · {self.surface} · RAS")

    def _pick(self, point, clicked):
        if self.atlas is None or not hasattr(self, "_triangles") or self._geometry_dirty:
            return
        inverse, valid = self._matrices()[0].inverted()
        if not valid:
            return
        x = 2 * point.x() / max(1, self.width()) - 1
        y = 1 - 2 * point.y() / max(1, self.height())
        ends = [inverse.map(QVector3D(x, y, z)) for z in (-1, 1)]
        origin = np.array(ends[0].toTuple())
        direction = np.array(ends[1].toTuple()) - origin
        direction /= np.linalg.norm(direction)
        visible = np.flatnonzero(self._visible())
        index = pick_triangle(origin, direction, self._triangles[self.surface][visible])
        parcel = -1 if index is None else int(self.face_parcels[visible[index]])
        self.identified.emit(parcel, clicked)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press = self._last = event.position()
            self._dragged = False

    def mouseMoveEvent(self, event):
        if self._press is not None:
            delta = event.position() - self._last
            self._dragged |= (event.position() - self._press).manhattanLength() > 3
            self.yaw += delta.x() * .5
            self.pitch += delta.y() * .5
            self._last = event.position()
            self.update()
        elif time.monotonic() - self._last_hover > .1:
            self._last_hover = time.monotonic()
            self._pick(event.position(), False)

    def mouseReleaseEvent(self, event):
        if self._press is not None and not self._dragged:
            self._pick(event.position(), True)
        self._press = None

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * 1.1 ** (event.angleDelta().y() / 120), .4, 3.))
        self.update()
        event.accept()
