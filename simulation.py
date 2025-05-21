#!/usr/bin/env python
import sys, random, argparse, os, datetime, time, re, logging, ssl, urllib.request, base64, io, warnings, dataclasses, copy, json, requests, math, xml.etree.ElementTree as ET
from datetime import timezone
from typing import List, Optional, Dict, Any, get_origin, get_args, Union
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
MAX_TILE_RANGE = 2
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

OSM_XML_DIR = "osm_data"
OSM_TILE_DIR = "osm_tiles"
TILE_ZOOM = 13

GOOGLE_MAP_TEXTURE = 'google_map.jpg'
EARTH_TEXTURE = 'earth.jpg'
EARTH_URLS = ['https://your-earth-texture-url.example/earth.jpg']

TILE_CACHE: Dict[str, int] = {}

def latlon_to_tile(lat: float, lon: float, z: int):
    n = 2 ** z
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return xtile, ytile

def download_tile(x: int, y: int, z: int):
    if not os.path.exists(OSM_TILE_DIR): os.makedirs(OSM_TILE_DIR)
    tile_file = os.path.join(OSM_TILE_DIR, f"{z}_{x}_{y}.png")
    if os.path.exists(tile_file): return tile_file
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    try:
        resp = requests.get(url, headers={'User-Agent': 'Sim'}, timeout=5, stream=True)
        if resp.status_code == 200:
            with open(tile_file, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
    except Exception as e:
        logging.error(f'Tile download failed: {e}')
    return tile_file

def generate_osm_map_texture(lat, lon, zoom, tile_range, out_file):
    x0, y0 = latlon_to_tile(lat, lon, zoom)
    size = 256 * (2 * tile_range + 1)
    try:
        map_img = Image.new('RGB', (size, size))
    except MemoryError:
        logging.error("Insufficient memory for tile range %d", tile_range)
        return
    for dy in range(-tile_range, tile_range + 1):
        row_img = Image.new('RGB', (size, 256))
        for dx in range(-tile_range, tile_range + 1):
            tx, ty = x0 + dx, y0 + dy
            tile_path = download_tile(tx, ty, zoom)
            try:
                with Image.open(tile_path) as tile_img:
                    row_img.paste(tile_img, ((dx + tile_range) * 256, 0))
            except Exception as e:
                logging.warning(f"Failed to process tile {tx},{ty}: {e}")
                continue
        map_img.paste(row_img, (0, (dy + tile_range) * 256))
    map_img.save(out_file)
    logging.info(f"Generated new map texture at {out_file}")

def convert_oem_to_png(oem_path: str, png_path: str):
    try:
        parser = ImageFile.Parser()
        with open(oem_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024), b''):
                parser.feed(chunk)
        img = parser.close()
        img.save(png_path)
        return png_path
    except Exception as e:
        logging.error(f'Failed to convert {oem_path} to PNG: {e}')
        return None

def ensure_texture(path: str, urls: List[str]) -> bool:
    if os.path.exists(path): return True
    for url in urls:
        try:
            resp = requests.get(url, headers={'User-Agent': 'Sim'}, timeout=5, stream=True)
            if resp.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                return True
        except Exception as e:
            logging.error(f'Failed to download texture from {url}: {e}')
    return False

for base in ('google_map', 'earth'):
    oem_file = f'{base}.oem'
    png_file = f'{base}.png'
    if os.path.exists(oem_file):
        new_tex = convert_oem_to_png(oem_file, png_file)
        if new_tex:
            if base == 'google_map': GOOGLE_MAP_TEXTURE = new_tex
            else: EARTH_TEXTURE = new_tex

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    class _PILStub:
        def __getattr__(self, _): raise RuntimeError("PIL not available")
    Image = _PILStub()
    class _PILImageFileStub: LOAD_TRUNCATED_IMAGES = False
    ImageFile = _PILImageFileStub()

from enum import Enum, auto
from dataclasses import dataclass, field

try:
    from OpenGL.GL import (
        glGenTextures, glBindTexture, glTexImage2D, glTexParameterf, glEnable, glDisable,
        glBegin, glEnd, glVertex3f, glVertex2f, glColor3f, glPointSize, glClear,
        glClearColor, glMatrixMode, glLoadIdentity, glPushMatrix, glPopMatrix,
        glRasterPos2f, glRasterPos3f, glTexCoord2f, glViewport,
        GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT, GL_PROJECTION, GL_MODELVIEW, GL_POINTS, GL_DEPTH_TEST,
        GL_VERTEX_ARRAY, GL_COLOR_ARRAY, GL_FLOAT, GL_MULTISAMPLE, GL_QUADS, GL_LINE_STRIP
    )
    from OpenGL.GLU import (
        gluPerspective, gluLookAt, gluNewQuadric, gluQuadricTexture,
        gluSphere, gluOrtho2D
    )
    from OpenGL.GLUT import (
        glutBitmapCharacter, GLUT_BITMAP_HELVETICA_10, GLUT_BITMAP_HELVETICA_12,
        glutInit, glutInitDisplayMode, glutInitWindowSize, glutCreateWindow,
        glutDisplayFunc, glutIdleFunc, glutKeyboardFunc, glutSpecialFunc, glutMainLoop, glutSwapBuffers,
        glutLeaveMainLoop, glutPostRedisplay,
        GLUT_DOUBLE, GLUT_RGB, GLUT_DEPTH, GLUT_KEY_UP, GLUT_KEY_DOWN, GLUT_KEY_LEFT, GLUT_KEY_RIGHT
    )
    GL_TRUE = 1
except Exception as e:
    logging.warning(f'OpenGL import failed: {e}')
    def _stub(*_, **__): pass
    glGenTextures = glBindTexture = glTexImage2D = glTexParameterf = glEnable = glDisable = _stub
    glBegin = glEnd = glVertex3f = glVertex2f = glColor3f = glPointSize = _stub
    glClear = glClearColor = glMatrixMode = glLoadIdentity = glPushMatrix = glPopMatrix = _stub
    glRasterPos2f = glRasterPos3f = glTexCoord2f = _stub
    glEnableClientState = glDisableClientState = glVertexPointer = glColorPointer = _stub
    GL_TEXTURE_2D = GL_RGB = GL_UNSIGNED_BYTE = GL_TEXTURE_MIN_FILTER = GL_TEXTURE_MAG_FILTER = GL_LINEAR = 0
    GL_COLOR_BUFFER_BIT = GL_DEPTH_BUFFER_BIT = GL_PROJECTION = GL_MODELVIEW = GL_POINTS = GL_DEPTH_TEST = GL_QUADS = GL_LINE_STRIP = 0
    GL_VERTEX_ARRAY = GL_COLOR_ARRAY = GL_FLOAT = GL_MULTISAMPLE = 0
    GL_TRUE = 1
    def gluPerspective(*_, **__): pass
    def gluLookAt(*_, **__): pass
    def gluNewQuadric(): return None
    def gluQuadricTexture(*_, **__): pass
    def gluSphere(*_, **__): pass
    def gluOrtho2D(*_, **__): pass
    def glutBitmapCharacter(*_, **__): pass
    GLUT_BITMAP_HELVETICA_10 = GLUT_BITMAP_HELVETICA_12 = None
    def glutInit(*_): pass
    def glutInitDisplayMode(*_, **__): pass
    def glutInitWindowSize(*_, **__): pass
    def glutCreateWindow(*_, **__): pass
    def glutDisplayFunc(*_, **__): pass
    def glutIdleFunc(*_, **__): pass
    def glutKeyboardFunc(*_, **__): pass
    def glutSpecialFunc(*_, **__): pass
    def glutMainLoop(*_, **__): pass
    def glutSwapBuffers(*_, **__): pass
    def glutLeaveMainLoop(*_, **__): pass
    def glutPostRedisplay(*_, **__): pass
    GLUT_DOUBLE = GLUT_RGB = GLUT_DEPTH = 0
    GLUT_KEY_UP = GLUT_KEY_DOWN = GLUT_KEY_LEFT = GLUT_KEY_RIGHT = None

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

G = 9.81
RAD = np.pi / 180.0
FT_PER_M = 3.28084
MPH_PER_KMH = 0.621371
FIRE_RANGE = 30000.0
EARTH_RADIUS_M = 6371000.0
GROUND_FRICTION = 9.0
TAXI_DISTANCE_M = 50.0

RUNWAY_LENGTH_M = 50.0
RUNWAY_TIME_S = None
AIRPORTS = {'KIAH': (29.9902, -95.3368), 'KSFO': (37.6189, -122.3750)}
RUNWAYS = {
    'KIAH': [{'hdg': 150.0, 'length': RUNWAY_LENGTH_M}],
    'KSFO': [{'hdg': 100.0, 'length': RUNWAY_LENGTH_M}]
}
_geo_cache: Dict[tuple, str] = {}

def get_location(lat: float, lon: float) -> str:
    key = (round(lat, 3), round(lon, 3))
    if key in _geo_cache: return _geo_cache[key]
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10&addressdetails=1'
    try:
        data = requests.get(url, headers={'User-Agent': 'Sim'}, timeout=2).json()
        addr = data.get('address', {})
        city = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('hamlet') or ''
        state = addr.get('state') or ''
        country = addr.get('country') or ''
        loc = ', '.join([s for s in (city, state, country) if s])
    except Exception:
        loc = ''
    _geo_cache[key] = loc
    return loc

def _ensure_defaults(obj):
    if obj is None: return
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is None:
                if f.default is not dataclasses.MISSING:
                    setattr(obj, f.name, copy.deepcopy(f.default))
                elif f.default_factory is not dataclasses.MISSING:
                    setattr(obj, f.name, f.default_factory())
                else:
                    t = f.type
                    origin = get_origin(t) or t
                    if dataclasses.is_dataclass(origin):
                        setattr(obj, f.name, origin())
                    elif origin is Union:
                        u_args = [a for a in get_args(t) if dataclasses.is_dataclass(a)]
                        setattr(obj, f.name, u_args[0]() if u_args else None)
                    elif origin in (float, int, bool, str):
                        setattr(obj, f.name, origin())
                    elif origin in (list, dict, set, tuple):
                        setattr(obj, f.name, origin())
                    else:
                        setattr(obj, f.name, None)
                v = getattr(obj, f.name)
            _ensure_defaults(v)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj: _ensure_defaults(item)
    elif isinstance(obj, dict):
        for item in obj.values(): _ensure_defaults(item)

class RealisticWeather:
    def __init__(self, ws=None, wd=None, seed=1):
        random.seed(seed)
        self.ws = ws if ws is not None else random.uniform(0, 20)
        self.wd = wd if wd is not None else random.uniform(0, 360)

class _SparseNN:
    def __init__(self, threshold: float = 0.01): self.threshold = threshold
    def map(self, _k: str, value: Any) -> Any:
        if value is None: return 1.0
        if isinstance(value, (int, float)) and not math.isfinite(value): return 1.0
        return value

class _InitMixin:
    def initialize(self, data: Dict[str, Any], snn: Optional[_SparseNN] = None):
        snn = snn or _SparseNN()
        for k, v in (data or {}).items():
            if hasattr(self, k): setattr(self, k, snn.map(k, v))
    def __post_init__(self):
        self.initialize({}); _ensure_defaults(self)

@dataclass
class GeoSpatialState(_InitMixin):
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    speed_mps: float = 0.0

@dataclass
class OrientationState(_InitMixin):
    phi: float = 0.0
    theta: float = 0.0
    psi: float = 0.0
    p:float=0.0; q:float=0.0; r:float=0.0

@dataclass
class PhysicalState(_InitMixin):
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0
    acceleration: float = 0.0

@dataclass
class MaterialState(_InitMixin):
    temperature: float = 15.0
    pressure: float = 0.0
    humidity: float = 0.0
    integrity: float = 1.0

@dataclass
class DecisionMakingState(_InitMixin):
    mode: str = "IDLE"
    target: Optional[str] = None
    threat_level: float = 0.0

@dataclass
class ElectronicState(_InitMixin):
    cpu_load: float = 0.0
    gpu_load: float = 0.0
    battery_pct: float = 1.0
    network_latency_ms: float = 0.0

@dataclass
class AlgorithmicState(_InitMixin):
    model_name: str = ""
    version: str = ""
    confidence: float = 0.0
    loss: float = 0.0

@dataclass
class MagneticFieldState(_InitMixin):
    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0

@dataclass
class EMRadiationState(_InitMixin):
    e_field_vpm: float = 0.0
    power_density_wpm2: float = 0.0
    frequency_hz: float = 0.0

@dataclass
class RadarSignatureState(_InitMixin):
    rcs_m2: float = 0.0
    detection_range_m: float = 0.0
    cone_deg: float = 60.0
    detection_coeff: float = 1.0
    active: bool = True

@dataclass
class NeuralNetworkState(_InitMixin):
    layers: List[str] = field(default_factory=list)
    sparse_activation: bool = True
    threshold: float = 0.01

@dataclass
class FlightControlState(_InitMixin):
    speed: float = 0.0
    throttle: float = 0.0
    on_ground: bool = True
    destroyed: bool = False
    fuel_mass: float = 15000.0
    fuel_capacity: float = 15000.0
    missile_total: int = 0
    missile_ready: int = 0

@dataclass
class SimulationState(_InitMixin):
    elapsed: float = 0.0
    sim_speed: float = 1.0

@dataclass
class EntityState(_InitMixin):
    geospatial: GeoSpatialState = field(default_factory=GeoSpatialState)
    orientation: OrientationState = field(default_factory=OrientationState)
    physical: PhysicalState = field(default_factory=PhysicalState)
    material: MaterialState = field(default_factory=MaterialState)
    decision: DecisionMakingState = field(default_factory=DecisionMakingState)
    electronic: ElectronicState = field(default_factory=ElectronicState)
    algorithmic: AlgorithmicState = field(default_factory=AlgorithmicState)
    magnetic: MagneticFieldState = field(default_factory=MagneticFieldState)
    em: EMRadiationState = field(default_factory=EMRadiationState)
    radar: RadarSignatureState = field(default_factory=RadarSignatureState)
    nn: NeuralNetworkState = field(default_factory=NeuralNetworkState)
    flight: FlightControlState = field(default_factory=FlightControlState)

    def initialize(self, data: Dict[str, Any], snn: Optional[_SparseNN] = None):
        snn = snn or _SparseNN()
        for k, v in (data or {}).items():
            if hasattr(self, k):
                attr = getattr(self, k)
                if dataclasses.is_dataclass(attr): attr.initialize(v, snn)
                else: setattr(self, k, snn.map(k, v))
        _ensure_defaults(self)

    def _step_generic(self, parent, dt):
        if self.flight.destroyed: return
        throttle = getattr(parent, 'pilot_input', PilotInput()).throttle
        if self.flight.on_ground and not getattr(parent, 'takeoff_authorized', False) and getattr(parent, 'taxi_remaining', TAXI_DISTANCE_M) <= 0.0:
            throttle = 0.0
        accel = throttle * 30.0 - self.flight.throttle * 10.0
        if self.flight.on_ground and throttle < 0.05 and self.flight.speed > 0.0:
            accel -= GROUND_FRICTION
        self.physical.acceleration = accel
        self.flight.speed = np.maximum(0.0, self.flight.speed + accel * dt)
        dist = self.flight.speed * dt
        if self.flight.on_ground and hasattr(parent, 'taxi_remaining') and parent.taxi_remaining > 0.0:
            parent.taxi_remaining = np.maximum(0.0, parent.taxi_remaining - dist)
        hdg = self.orientation.psi
        dlat = (dist / 111320.0) * np.cos(hdg)
        denom = 111320.0 * np.maximum(1e-3, np.cos(np.radians(self.geospatial.lat)))
        dlon = (dist / denom) * np.sin(hdg)
        self.geospatial.lat += dlat
        self.geospatial.lon += dlon
        climb = getattr(parent, "climb_rate", 0.0)
        self.geospatial.alt_m = np.maximum(0.0, self.geospatial.alt_m + climb * dt * (0 if self.flight.on_ground else 1))

    def step_aircraft(self, parent, dt): self._step_generic(parent, dt)
    def step_car(self, parent, dt): self._step_generic(parent, dt)

    def step_orbit(self, parent, dt):
        t = (UNIVERSE.simulation.elapsed + getattr(parent, "phase", 0.0)) % parent.orbit_period
        ang = 2 * np.pi * t / parent.orbit_period
        self.geospatial.lat = 0.0
        self.geospatial.lon = np.degrees(ang)
        self.geospatial.alt_m = parent.orbit_alt_km * 1000.0
        self.flight.on_ground = False

    def step_launch(self, parent, dt):
        if self.flight.destroyed: return
        if parent.launch_phase == 0:
            self.physical.acceleration = 15.0 * G
            self.flight.speed += self.physical.acceleration * dt
            self.geospatial.alt_m += self.flight.speed * dt
            if self.geospatial.alt_m >= 80000.0: parent.launch_phase = 1
        else:
            self.physical.acceleration = 0.0
            self.flight.speed = np.maximum(0.0, self.flight.speed - G * dt)

    def euler_deg(self):
        return (
            self.orientation.theta / RAD,
            self.orientation.phi / RAD,
            ((self.orientation.psi + np.pi) % (2 * np.pi) - np.pi) / RAD
        )

State6DOF = EntityState

@dataclass
class UniversalBo(_InitMixin):
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(timezone.utc))
    simulation: SimulationState = field(default_factory=SimulationState)
    entities: Dict[str, EntityState] = field(default_factory=dict)

    def add_entity(self, name: str, state: Optional[EntityState] = None) -> EntityState:
        if name not in self.entities: self.entities[name] = state or EntityState()
        self.entities[name].initialize({})
        self.validate_required_fields()
        return self.entities[name]

    def get(self, name: str) -> Optional[EntityState]:
        return self.entities.get(name)

    def validate_required_fields(self):
        for e in self.entities.values(): _ensure_defaults(e)
        _ensure_defaults(self.simulation)

UNIVERSE = UniversalBo()

def calc_heading(orig_lat, orig_lon, dest_lat, dest_lon):
    dLon = np.radians(dest_lon - orig_lon)
    y = np.sin(dLon) * np.cos(np.radians(dest_lat))
    x = np.cos(np.radians(orig_lat)) * np.sin(np.radians(dest_lat)) - np.sin(np.radians(orig_lat)) * np.cos(np.radians(dest_lat)) * np.cos(dLon)
    return (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def magnetic_variation(lat, lon): return 10.0 * np.sin(np.radians(lon)) * np.cos(np.radians(lat))
def geo_to_xyz(lat, lon, radius=1.0):
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    cos_lat = np.cos(lat_r)
    x = radius * cos_lat * np.cos(lon_r)
    y = radius * np.sin(lat_r)
    z = radius * cos_lat * np.sin(lon_r)
    return x, y, z

def runway_designation(hdg): return int(round(hdg / 10.0)) % 36 or 36

class Sensors:
    def __init__(self): self.aoa_L = self.aoa_R = self.mach = self.flap = self.on_gnd = self.fail_L = False
    def set_env(self, base_aoa, mach, flap, on_gnd):
        self.aoa_L = self.aoa_R = base_aoa
        self.mach, self.flap, self.on_gnd = mach, flap, on_gnd
    def aoa_status(self): return self.fail_L
    def avg_aoa(self): return 0.5 * (self.aoa_L + self.aoa_R)

@dataclass
class PilotInput:
    throttle: float = 0.0
    yoke: float = 0.0
    roll: float = 0.0
    rudder: float = 0.0
    trim: float = 0.0
    flap_cmd: int = 0
    gear_cmd: bool = False

@dataclass
class APCommand:
    yoke: float = 0.0
    roll: float = 0.0

class Autopilot:
    def __init__(self):
        self.tgt_alt = 0.0
        self.tgt_hdg = 0.0
    def set_targets(self, alt, hdg): self.tgt_alt, self.tgt_hdg = alt, hdg
    def command(self, _dt, _state): return APCommand()

class PilotModel:
    def decide(self, ac, flights, missiles, tankers):
        if ac.state.flight.on_ground:
            if not ac.takeoff_authorized:
                if hasattr(ac, 'takeoff_line_lat'):
                    d = haversine(ac.state.geospatial.lat, ac.state.geospatial.lon, ac.takeoff_line_lat, ac.takeoff_line_lon) * 1000.0
                else:
                    d = 0.0
                if d > 10.0: return (0.0, ac.runway_hdg, 0.2)
                if ac.state.flight.speed < 0.5: return (0.0, ac.runway_hdg, 0.0)
                return (0.0, ac.runway_hdg, 0.0)
            return (1000.0, ac.runway_hdg, 1.0)
        inbound = [m for m in missiles if m['target'] is ac and m['dist'] < 8000.0]
        if inbound:
            ev_dir = (ac.state.orientation.psi + np.pi / 2) % (2 * np.pi)
            return (15000.0, ev_dir, 1.0)
        enemies = [f for f in flights if f.team != ac.team and isinstance(f, Fighter) and not f.state.flight.destroyed]
        if enemies:
            target = min(enemies, key=lambda t: haversine(ac.state.geospatial.lat, ac.state.geospatial.lon, t.state.geospatial.lat, t.state.geospatial.lon))
            hdg = calc_heading(ac.state.geospatial.lat, ac.state.geospatial.lon, target.state.geospatial.lat, target.state.geospatial.lon)
            return (12000.0, hdg, 1.0)
        return (8000.0, ac.state.orientation.psi, 0.8)

class BaseAgent:
    DEFAULT_ATTRS_SIMPLE = {'climb_rate': 0.0, 'turn_rate': 0.0, 'takeoff_authorized': False, 'runway_hdg': 0.0}
    def __init__(self, name: str):
        self.name = name
        self.state = UNIVERSE.add_entity(name)
        if not hasattr(self, 'pilot_input'): self.pilot_input = PilotInput()
        for attr, default in BaseAgent.DEFAULT_ATTRS_SIMPLE.items():
            if not hasattr(self, attr): setattr(self, attr, default)
        self._failsafe_init()
    def _failsafe_init(self):
        _ensure_defaults(self)
        _ensure_defaults(self.state)

class Missile(BaseAgent):
    def __init__(self, name, shooter=None, missile_type='GENERIC'):
        super().__init__(name)
        self.missile_type = missile_type
        self.active = False
        self.speed = 800.0
        self.shooter = shooter
        self.status = 'READY'
        if shooter is not None:
            self.state.geospatial.lat = shooter.state.geospatial.lat
            self.state.geospatial.lon = shooter.state.geospatial.lon
            self.state.geospatial.alt_m = shooter.state.geospatial.alt_m
            self.state.orientation.psi = shooter.state.orientation.psi
            self.state.flight.speed = shooter.state.flight.speed
            self.state.flight.on_ground = shooter.state.flight.on_ground
        else:
            self.state.flight.speed = self.speed
            self.state.flight.on_ground = True
        self._failsafe_init()
    def launch(self):
        self.active = True
        self.status = 'IN_FLIGHT'
    def step(self, dt):
        if not self.active: return
        self.state.geospatial.alt_m = np.maximum(0.0, self.state.geospatial.alt_m + 200.0 * dt)

class Aircraft(BaseAgent):
    def __init__(self, callsign='AC', team='EAST'):
        super().__init__(callsign)
        self.callsign = callsign
        self.ap = Autopilot()
        self.pilot = PilotModel()
        self.team = team
        self.dest = None
        self.rtb = False
        self.climb_rate = getattr(self, 'climb_rate', 50.0)
        if self.climb_rate == 0.0: self.climb_rate = 50.0
        self.turn_rate = getattr(self, 'turn_rate', np.radians(10.0))
        self.missiles: List[Missile] = []
        self.prev_psi = 0.0
        self.route = []
        self.route_idx = 0
        self.runway_hdg = getattr(self, 'runway_hdg', 0.0)
        self.pilot_name = ""
        self.missile_type = 'GENERIC'
        self.takeoff_authorized = getattr(self, 'takeoff_authorized', False)
        self.liftoff_speed = 80.0
        self.pilot_input = PilotInput()
        self.v1 = self.liftoff_speed * 0.7
        self.v2 = self.liftoff_speed * 0.9
        self.v1_time: Optional[float] = None
        self.v2_time: Optional[float] = None
        self.runway_distance = 0.0
        self.takeoff_roll_start_time = None
        self.taxi_remaining = TAXI_DISTANCE_M
        self._failsafe_init()
    def _update_runway_tracking(self, dt):
        global RUNWAY_TIME_S
        if not self.takeoff_authorized: return
        if self.state.flight.on_ground:
            if self.takeoff_roll_start_time is None and self.state.flight.speed > 0:
                self.takeoff_roll_start_time = UNIVERSE.simulation.elapsed
                self.runway_distance = 0.0
            if self.takeoff_roll_start_time is not None:
                self.runway_distance += self.state.flight.speed * dt
                if RUNWAY_TIME_S is None and self.runway_distance >= RUNWAY_LENGTH_M:
                    RUNWAY_TIME_S = UNIVERSE.simulation.elapsed - self.takeoff_roll_start_time
        else:
            self.takeoff_roll_start_time = None
    def step(self, dt):
        self.state.step_aircraft(self, dt)
        self._update_runway_tracking(dt)
    def load_missiles(self, count, missile_type=None):
        self.missiles = []
        m_type = missile_type if missile_type else self.missile_type
        for i in range(1, count + 1):
            self.missiles.append(Missile(f'{self.callsign}_MSL{i}', self, m_type))
        self.state.flight.missile_total = count
        self.state.flight.missile_ready = count
    def fire_missile(self):
        for m in self.missiles:
            if m.status == 'READY':
                m.launch()
                self.state.flight.missile_ready -= 1
                return m
        return None

class Fighter(Aircraft):
    def __init__(self, callsign='FTR', team='EAST'):
        super().__init__(callsign, team)
        if not self.missiles:
            self.load_missiles(getattr(self, 'missile_count', 0), getattr(self, 'missile_type', 'GENERIC'))
        self._failsafe_init()

class Refueler(Aircraft): pass

class F35(Fighter):
    MAX_GROUND_SPEED = 90.0
    MAX_MIL_POWER_SPEED = 320.0
    MAX_SUPERCRUISE_SPEED = 408.0
    MAX_AFTERBURNER_SPEED = 540.0
    def __init__(self, callsign='F35', team='WEST'):
        super().__init__(callsign, team)
        self.liftoff_speed = 85.0
        self.v1 = 70.0
        self.v2 = 85.0
    def step(self, dt):
        super().step(dt)
        if self.state.flight.on_ground and self.state.flight.speed >= self.liftoff_speed:
            self.state.flight.on_ground = False
        if self.state.flight.on_ground:
            if self.state.flight.speed > self.MAX_GROUND_SPEED:
                self.state.flight.speed = self.MAX_GROUND_SPEED
        else:
            if self.state.flight.throttle > 0.9:
                cap = self.MAX_AFTERBURNER_SPEED
            elif self.state.flight.throttle > 0.7:
                cap = self.MAX_SUPERCRUISE_SPEED
            else:
                cap = self.MAX_MIL_POWER_SPEED
            if self.state.flight.speed > cap:
                self.state.flight.speed = cap

class F22(Fighter): pass
class KC46(Refueler): pass

class Car(Aircraft):
    def step(self, dt): self.state.step_car(self, dt)
class Airliner(Aircraft): pass

class TeslaCar(Car):
    def __init__(self, callsign, driver_name, route):
        super().__init__(callsign, 'CIV')
        self.driver_name = driver_name
        self.route = route
        self.coverage = True
        self.speed = 30.0 / 3.6
        self._failsafe_init()

class Boeing737Max(Airliner):
    def __init__(self, callsign='B737', team='CIV'):
        super().__init__(callsign, team)
        self.mcas = type('MCAS', (), {'cmd_count': 0, 'total_cmd': 0.0})()
        self.mcas_reported = False
        self._failsafe_init()
class Boeing737MaxV1(Boeing737Max): pass
class Boeing737MaxV2(Boeing737Max): pass

class OrbitalAsset(Aircraft):
    def __init__(self, callsign='ORB', alt_km=420):
        super().__init__(callsign, 'SPACE')
        self.orbit_alt_km = alt_km
        self.orbit_radius = 6371 + alt_km
        self.orbit_period = 92 * 60
        self.phase = random.random() * 2 * np.pi
        self.state.flight.on_ground = False
        self._failsafe_init()
    def step(self, dt): self.state.step_orbit(self, dt)

class ISS(OrbitalAsset):
    def __init__(self): super().__init__('ISS', 415)
class StarlinkSatellite(OrbitalAsset):
    def __init__(self, idx):
        super().__init__(f'STAR{idx}', 550)
        self.phase = idx * 30 * RAD
        self._failsafe_init()

class LaunchVehicle(Aircraft):
    def __init__(self, callsign='LV'):
        super().__init__(callsign, 'SPACE')
        self.launch_phase = 0
        self.state.geospatial.alt_m = 0.0
        self._failsafe_init()
    def step(self, dt): self.state.step_launch(self, dt)

class Falcon9(LaunchVehicle):
    def __init__(self): super().__init__('FALCON9'); self.mission = 'ISS_REFUEL'; self._failsafe_init()
class FalconHeavy(LaunchVehicle): pass
class BoeingSpaceliner(LaunchVehicle): pass

class BrandDataLoader:
    BRAND_DATA: Dict[str, Dict[str, Any]] = {
        'F35': {'missile_count': 4, 'missile_type': 'AIM-260', 'radar_detection_m': 150000, 'rcs_m2': 0.001, 'radar_detection_coeff': 1.0},
        'F22': {'missile_count': 4, 'missile_type': 'AIM-120D', 'radar_detection_m': 140000, 'rcs_m2': 0.0001, 'radar_detection_coeff': 1.0}
    }
    @staticmethod
    def get(brand: str) -> Dict[str, Any]:
        return BrandDataLoader.BRAND_DATA.get(brand, {}).copy()

FLIGHTS = [
    {'type': 'F35', 'callsign': 'BO1', 'airport': 'KIAH', 'team': 'WEST', 'pilot_name': 'Maj. Bo', **BrandDataLoader.get('F35')},
    {'type': 'F35', 'callsign': 'MACE2', 'airport': 'KIAH', 'team': 'WEST', 'pilot_name': 'Capt. Alex "Mace" Knight', **BrandDataLoader.get('F35')},
    {'type': 'F22', 'callsign': 'VIPER1', 'airport': 'KSFO', 'team': 'EAST', 'pilot_name': 'Capt. Jay "Ghost" Lee', **BrandDataLoader.get('F22')},
    {'type': 'F22', 'callsign': 'VIPER2', 'airport': 'KSFO', 'team': 'EAST', 'pilot_name': 'Capt. Mia "Storm" Kim', **BrandDataLoader.get('F22')},
    {'type': 'KC46', 'callsign': 'TANK1', 'airport': 'KIAH', 'team': 'WEST'},
    {'type': 'B737', 'callsign': 'CIV1', 'airport': 'KIAH', 'team': 'CIV'}
]

WAR_GAMES_CONFIG = {'sim_dt': 0.02, 'render_fps': 45, 'flights': FLIGHTS}

def load_texture(path):
    try:
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        img = Image.open(path).convert("RGB")
        img_data = img.tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex_id
    except Exception as e:
        logging.error(f'Texture load failed: {e}')
        return 0

def load_tile_texture(path):
    try:
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        img = Image.open(path).convert("RGB")
        data = img.tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return tex_id
    except Exception as e:
        logging.error(f'Tile texture load failed: {e}')
        return 0

def draw_text(x, y, z, text):
    glRasterPos3f(x, y, z)
    for ch in text: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, ord(ch))

class Simulation:
    MAP_ZOOM = 19

    def __init__(self, cfg, duration=3600.0, seed=1, view_only=False):
        self.cfg = cfg
        self.duration = duration
        self.view_only = view_only
        self.weather = RealisticWeather()
        self.flights: List[Aircraft] = []
        self.tankers: List[KC46] = []
        self.missiles = []
        self.bo_ac: Optional[Aircraft] = None
        self.window_width, self.window_height = 1200, 800
        self.map_zoom = Simulation.MAP_ZOOM
        self.cam_yaw_offset = 0.0
        self.cam_pitch_offset = 0.0
        self.idle_last = time.time()
        self.view_mode = 'MAP'
        self.selected_index = 0
        self._init_flights(seed)
        self.google_ok = os.path.exists(GOOGLE_MAP_TEXTURE)
        self.earth_ok = ensure_texture(EARTH_TEXTURE, EARTH_URLS)
        self.tex_id = 0

    def load_osm_data(self):
        if not self.bo_ac: return
        lat0, lon0 = self.bo_ac.state.geospatial.lat, self.bo_ac.state.geospatial.lon
        delta = 0.01
        minlat, minlon, maxlat, maxlon = lat0 - delta, lon0 - delta, lat0 + delta, lon0 + delta
        if not os.path.exists(OSM_XML_DIR): os.makedirs(OSM_XML_DIR)
        fname = f"map_{minlon:.6f}_{minlat:.6f}_{maxlon:.6f}_{maxlat:.6f}.xml"
        path = os.path.join(OSM_XML_DIR, fname)
        if not os.path.exists(path):
            url = f"https://api.openstreetmap.org/api/0.6/map?bbox={minlon},{minlat},{maxlon},{maxlat}"
            resp = requests.get(url, headers={'User-Agent':'Sim'}, timeout=10)
            with open(path, 'wb') as f: f.write(resp.content)
        self.osm_nodes = {}
        self.osm_ways = []
        for event, elem in ET.iterparse(path, events=('end',)):
            if elem.tag == 'node':
                nid = elem.get('id')
                self.osm_nodes[nid] = (float(elem.get('lat')), float(elem.get('lon')))
                elem.clear()
            elif elem.tag == 'way':
                coords = []
                for nd in elem.findall('nd'):
                    ref = nd.get('ref')
                    if ref in self.osm_nodes: coords.append(self.osm_nodes[ref])
                if coords: self.osm_ways.append(coords)
                elem.clear()
        logging.debug(f"Parsed {len(self.osm_nodes)} nodes and {len(self.osm_ways)} ways from OSM XML")

    def _init_flights(self, seed):
        random.seed(seed)
        self.flights = []
        self.tankers = []
        self.missiles = []
        self.takeoff_schedule = {}
        self.takeoff_cleared = {}
        for fconf in self.cfg.get('flights', []):
            typ = fconf.get('type')
            callsign = fconf.get('callsign')
            team = fconf.get('team')
            airport = fconf.get('airport')
            cls = globals().get(typ)
            if typ == 'B737':
                cls = Boeing737Max
            if cls is None:
                continue
            ac = cls(callsign=callsign, team=team)
            for attr in ('pilot_name', 'missile_count', 'missile_type', 'radar_detection_m', 'rcs_m2', 'radar_detection_coeff'):
                if attr in fconf:
                    setattr(ac, attr, fconf[attr])
            if hasattr(ac, 'load_missiles') and 'missile_count' in fconf:
                ac.load_missiles(fconf['missile_count'], fconf.get('missile_type'))
            lat, lon = AIRPORTS.get(airport, (0.0, 0.0))
            ac.state.geospatial.lat = lat
            ac.state.geospatial.lon = lon
            runways = RUNWAYS.get(airport, [])
            hdg = runways[0]['hdg'] if runways else 0.0
            ac.state.orientation.psi = hdg * RAD
            ac.runway_hdg = hdg * RAD
            if isinstance(ac, KC46):
                self.tankers.append(ac)
            else:
                self.flights.append(ac)
            t_auth = random.uniform(1.0, 10.0)
            self.takeoff_schedule[callsign] = t_auth
            self.takeoff_cleared[callsign] = False
        if self.flights:
            self.bo_ac = self.flights[0]

    def run(self):
        try:
            glutInit(sys.argv)
            glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
            glutInitWindowSize(self.window_width, self.window_height)
            glutCreateWindow(b"Bo War Game Sim")
            self.quadric = gluNewQuadric()
            gluQuadricTexture(self.quadric, GL_TRUE)
            self.update_texture()
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_MULTISAMPLE)
            glClearColor(0.05, 0.05, 0.1, 1.0)
            renderer = SimulationRenderer(self)
            glutDisplayFunc(renderer.display)
            glutIdleFunc(renderer.idle)
            glutKeyboardFunc(renderer.keyboard)
            glutSpecialFunc(renderer.special)
            glutMainLoop()
        except Exception as e:
            logging.error(f'OpenGL init failed: {e}')

    def update_texture(self):
        if self.tex_id != 0:
            from OpenGL.GL import glDeleteTextures
            glDeleteTextures([self.tex_id])
            self.tex_id = 0
        texture_path = GOOGLE_MAP_TEXTURE if self.view_mode == 'MAP' else EARTH_TEXTURE
        self.tex_id = load_texture(texture_path)

class SimulationRenderer:
    def __init__(self, sim: Simulation):
        self.sim = sim

    def display(self):
        glViewport(0, 0, self.sim.window_width, self.sim.window_height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.sim.view_mode == '3D':
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            gluPerspective(45.0, self.sim.window_width/self.sim.window_height, 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            self.sim.set_camera()
            if self.sim.tex_id:
                glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.sim.tex_id)
            else:
                glColor3f(0.2, 0.6, 1.0)
            gluSphere(self.sim.quadric, 1.0, 64, 64)
            if self.sim.tex_id: glDisable(GL_TEXTURE_2D)
            self.draw_flights()
        else:
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            gluOrtho2D(0, self.sim.window_width, 0, self.sim.window_height)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            if self.sim.tex_id:
                glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.sim.tex_id)
                glBegin(GL_QUADS)
                glTexCoord2f(0,0); glVertex2f(0,0)
                glTexCoord2f(1,0); glVertex2f(self.sim.window_width,0)
                glTexCoord2f(1,1); glVertex2f(self.sim.window_width,self.sim.window_height)
                glTexCoord2f(0,1); glVertex2f(0,self.sim.window_height)
                glEnd(); glDisable(GL_TEXTURE_2D)
            self.draw_osm_features()
        self.draw_overlay()
        glutSwapBuffers()

    def draw_osm_features(self):
        if not hasattr(self.sim, 'osm_ways'): return
        glColor3f(0.0,1.0,0.0)
        for way in self.sim.osm_ways:
            glBegin(GL_LINE_STRIP)
            for lat, lon in way:
                x,y = self.latlon_to_screen(lat, lon)
                glVertex2f(x,y)
            glEnd()

    def draw_flights(self):
        if not self.sim.flights: return
        glPointSize(8.0); glColor3f(0.0,0.0,1.0); glBegin(GL_POINTS)
        for ac in self.sim.flights:
            lat, lon = ac.state.geospatial.lat, ac.state.geospatial.lon
            alt_factor = 1.02 + max(0.0, ac.state.geospatial.alt_m)/EARTH_RADIUS_M
            x,y,z = geo_to_xyz(lat, lon, alt_factor)
            glVertex3f(x,y,z)
        glEnd()
        for ac in self.sim.flights:
            lat, lon = ac.state.geospatial.lat, ac.state.geospatial.lon
            alt_factor = 1.02 + max(0.0, ac.state.geospatial.alt_m)/EARTH_RADIUS_M
            x,y,z = geo_to_xyz(lat, lon, alt_factor)
            loc = get_location(lat, lon)
            draw_text(x, y+0.02, z, f"{ac.callsign} {loc}")

    def draw_overlay(self):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, self.sim.window_width, 0, self.sim.window_height)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glColor3f(1,1,1)
        planes = [ac for ac in self.sim.flights if isinstance(ac, (F35, F22))]
        east = [p for p in planes if p.team=='EAST'][:2]
        west = [p for p in planes if p.team=='WEST'][:2]
        selected = east + west
        corners = [(10, self.sim.window_height-50), (self.sim.window_width-250, self.sim.window_height-50), (10,150), (self.sim.window_width-250,150)]
        for ac,(x0,y0) in zip(selected, corners):
            vals = [
                f'{ac.callsign} {type(ac).__name__}',
                f'Alt {ac.state.geospatial.alt_m:.0f} m',
                f'Spd {ac.state.flight.speed*3.6:.0f} km/h',
                f'Hdg {np.degrees(ac.state.orientation.psi)%360:.0f}°',
                f'Fuel {ac.state.flight.fuel_mass/ac.state.flight.fuel_capacity*100:.0f}%'
            ]
            for i,l in enumerate(vals):
                glRasterPos2f(x0, y0-15*i)
                for ch in l: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

    def latlon_to_screen(self, lat, lon):
        z = self.sim.map_zoom; n = 2**z
        xtile_f = (lon+180.0)/360.0*n
        ytile_f = (1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat)))/math.pi)/2*n
        xtile, ytile = latlon_to_tile(lat, lon, z)
        dx = xtile_f-xtile; dy = ytile_f-ytile
        return dx*self.sim.window_width, dy*self.sim.window_height

    def idle(self):
        current = time.time()
        dt_real = current - self.sim.idle_last
        self.sim.idle_last = current
        if not self.sim.view_only: self.sim.sim_step(dt_real)
        glutPostRedisplay()

    def keyboard(self, key, _x, _y):
        if key==b'\x1b': sys.exit(0)
        elif key==b'm': self.sim.view_mode='MAP'; self.sim.update_texture()
        elif key==b'3': self.sim.view_mode='3D'; self.sim.update_texture()
        elif key==b'n': self.sim.selected_index+=1
        elif key==b'p': self.sim.selected_index-=1
        glutPostRedisplay()

    def special(self, key, x, y):
        if key==GLUT_KEY_LEFT: self.sim.cam_yaw_offset-=5*RAD
        elif key==GLUT_KEY_RIGHT: self.sim.cam_yaw_offset+=5*RAD
        elif key==GLUT_KEY_UP: self.sim.cam_pitch_offset=min(self.sim.cam_pitch_offset+5*RAD, np.radians(85))
        elif key==GLUT_KEY_DOWN: self.sim.cam_pitch_offset=max(self.sim.cam_pitch_offset-5*RAD, 0)
        glutPostRedisplay()

def run_war_games(dt=0.02, duration=3600.0, seed=1, view_only=False):
    cfg = WAR_GAMES_CONFIG
    cfg['sim_dt'] = dt
    sim = Simulation(cfg, duration=duration, seed=seed, view_only=True)
    sim.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--view_only", action="store_true")
    parser.add_argument("--war_games", action="store_true")
    args = parser.parse_args()
    if args.war_games:
        WAR_GAMES_CONFIG['sim_dt'] = args.dt
        sim = Simulation(WAR_GAMES_CONFIG, duration=args.duration, seed=args.seed, view_only=True)
        sim.run()
    else:
        cfg = WAR_GAMES_CONFIG
        cfg['sim_dt'] = args.dt
        sim = Simulation(cfg, duration=args.duration, seed=args.seed, view_only=args.view_only)
        sim.run()
