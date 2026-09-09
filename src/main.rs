//! A real-time ray tracer for the Sony PSP.
//!
//! Four spheres orbiting over an infinite checkerboard, with one reflection
//! bounce and hard shadows, traced per pixel on a 333 MHz MIPS Allegrex. The
//! PSP's GPU cannot do any of this: it is a fixed-function unit that textures
//! triangles, so every ray here is followed in software on the CPU.
//!
//! The scene is traced at half the screen's resolution in each direction and
//! each traced pixel is written as a 2x2 block, which fills 480x272 exactly.
//! Tracing every pixel would quadruple the work again for a screen this size.
//!
//! After `FRAMES` frames it writes ms0:/raytracer-result.json and exits, so
//! that psp-devloop can decide whether the run passed instead of a person
//! having to watch the screen.

#![no_std]
#![no_main]

use psp::sys;
use psp::{BUF_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH};

psp::module!("psp_raytracer", 1, 0);

/// Traced resolution. Scaled up by `SCALE` to exactly fill 480x272, so this is
/// half the screen in each direction: one ray per 2x2 block of pixels.
const RENDER_W: usize = 240;
const RENDER_H: usize = 136;
const SCALE: usize = 2;

/// Frames to render before writing the result file and exiting.
const FRAMES: u32 = 600;

/// How far a ray may travel before it is considered to have hit nothing.
const FAR: f32 = 1.0e9;
/// Offset along the normal when spawning secondary rays, so that a ray does
/// not immediately hit the surface it started from.
const EPS: f32 = 1.0e-3;

// ---------------------------------------------------------------- vectors

#[derive(Clone, Copy)]
struct V3 {
    x: f32,
    y: f32,
    z: f32,
}

const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

impl V3 {
    fn add(self, o: V3) -> V3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    fn sub(self, o: V3) -> V3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    fn scale(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
    fn dot(self, o: V3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    fn norm(self) -> V3 {
        let len2 = self.dot(self);
        if len2 <= 0.0 {
            return self;
        }
        self.scale(1.0 / libm::sqrtf(len2))
    }
    fn cross(self, o: V3) -> V3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    /// Mirror this direction about `n`, which must be normalised.
    fn reflect(self, n: V3) -> V3 {
        self.sub(n.scale(2.0 * self.dot(n)))
    }
}

// ------------------------------------------------------------------ scene

struct Sphere {
    center: V3,
    radius: f32,
    albedo: V3,
    /// 0.0 is matte, 1.0 is a mirror.
    mirror: f32,
}

/// Where the light sits. High and to one side, so the shadows fall across the
/// checkerboard where they can be seen.
const LIGHT: V3 = v3(4.0, 6.0, -3.0);
/// The checkerboard lies here; everything else floats above it.
const FLOOR_Y: f32 = -1.0;

/// What a ray ran into.
enum Hit {
    /// Index into the sphere array.
    Sphere(usize),
    Floor,
}

/// Builds the scene for a given time in seconds. Three coloured spheres orbit
/// a larger mirrored one, bobbing at different rates so the arrangement never
/// quite repeats.
fn scene(t: f32) -> [Sphere; 4] {
    let orbit = |phase: f32, radius: f32, speed: f32, bob: f32| -> V3 {
        let a = t * speed + phase;
        v3(
            libm::cosf(a) * radius,
            -0.45 + libm::sinf(t * bob + phase) * 0.35,
            libm::sinf(a) * radius + 0.4,
        )
    };

    [
        // The mirror at the centre, big enough to show the others in it.
        Sphere {
            center: v3(0.0, 0.15, 0.6),
            radius: 1.05,
            albedo: v3(0.92, 0.94, 0.97),
            mirror: 0.85,
        },
        Sphere {
            center: orbit(0.0, 2.1, 0.9, 1.7),
            radius: 0.5,
            albedo: v3(0.90, 0.24, 0.18),
            mirror: 0.25,
        },
        Sphere {
            center: orbit(2.094, 2.1, 0.9, 1.3),
            radius: 0.5,
            albedo: v3(0.20, 0.62, 0.90),
            mirror: 0.25,
        },
        Sphere {
            center: orbit(4.188, 2.1, 0.9, 2.1),
            radius: 0.5,
            albedo: v3(0.96, 0.78, 0.16),
            mirror: 0.25,
        },
    ]
}

/// Distance along `rd` at which the ray enters the sphere, if it does.
/// `rd` must be normalised.
fn hit_sphere(s: &Sphere, ro: V3, rd: V3) -> Option<f32> {
    let oc = ro.sub(s.center);
    let b = oc.dot(rd);
    let c = oc.dot(oc) - s.radius * s.radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let root = libm::sqrtf(disc);
    let near = -b - root;
    if near > EPS {
        return Some(near);
    }
    let far = -b + root;
    if far > EPS { Some(far) } else { None }
}

/// Nearest intersection with the whole scene.
fn trace(spheres: &[Sphere; 4], ro: V3, rd: V3) -> Option<(f32, Hit)> {
    let mut best = FAR;
    let mut what = None;

    for (i, s) in spheres.iter().enumerate() {
        if let Some(t) = hit_sphere(s, ro, rd)
            && t < best
        {
            best = t;
            what = Some(Hit::Sphere(i));
        }
    }

    // The floor is an infinite plane, so it needs no geometry at all: one
    // division says where the ray crosses y = FLOOR_Y.
    if rd.y < -1.0e-4 {
        let t = (FLOOR_Y - ro.y) / rd.y;
        if t > EPS && t < best {
            best = t;
            what = Some(Hit::Floor);
        }
    }

    what.map(|w| (best, w))
}

/// True when something stands between `p` and the light.
fn in_shadow(spheres: &[Sphere; 4], p: V3) -> bool {
    let to_light = LIGHT.sub(p);
    let dist = libm::sqrtf(to_light.dot(to_light));
    let dir = to_light.scale(1.0 / dist);

    spheres
        .iter()
        .any(|s| matches!(hit_sphere(s, p, dir), Some(t) if t < dist))
}

/// Colour of the sky in a given direction: a plain vertical gradient, which
/// also gives the mirrored sphere something to reflect.
fn sky(rd: V3) -> V3 {
    let t = (rd.y * 0.5 + 0.5).clamp(0.0, 1.0);
    v3(0.06, 0.09, 0.16).scale(1.0 - t).add(v3(0.35, 0.52, 0.85).scale(t))
}

/// Shade one hit point, without following any further rays.
fn shade(spheres: &[Sphere; 4], hit: &Hit, p: V3, n: V3) -> V3 {
    let albedo = match hit {
        Hit::Sphere(i) => spheres[*i].albedo,
        Hit::Floor => {
            // Checkerboard: the parity of the two floored coordinates.
            let cx = libm::floorf(p.x) as i32;
            let cz = libm::floorf(p.z) as i32;
            if (cx + cz) & 1 == 0 {
                v3(0.83, 0.83, 0.86)
            } else {
                v3(0.13, 0.14, 0.17)
            }
        }
    };

    let to_light = LIGHT.sub(p).norm();
    let mut light = 0.16; // ambient, so shadows are dark but not black
    if !in_shadow(spheres, p.add(n.scale(EPS * 8.0))) {
        let diffuse = n.dot(to_light).max(0.0);
        light += diffuse * 0.9;

        // A tight specular highlight, which is what makes them read as balls
        // rather than flat discs.
        let half = to_light.sub(v3(0.0, 0.0, 0.0)).norm();
        let spec = n.dot(half).max(0.0);
        light += spec * spec * spec * spec * spec * spec * spec * spec * 0.35;
    }

    albedo.scale(light)
}

/// Surface normal at a hit point.
fn normal_at(spheres: &[Sphere; 4], hit: &Hit, p: V3) -> V3 {
    match hit {
        Hit::Sphere(i) => p.sub(spheres[*i].center).norm(),
        Hit::Floor => v3(0.0, 1.0, 0.0),
    }
}

/// How mirrored the surface is.
fn mirror_of(spheres: &[Sphere; 4], hit: &Hit) -> f32 {
    match hit {
        Hit::Sphere(i) => spheres[*i].mirror,
        // A wet-looking floor; it also proves the reflection ray works, because
        // the spheres appear upside down in it.
        Hit::Floor => 0.22,
    }
}

/// Follow one primary ray and, where the surface is mirrored, one bounce.
fn radiance(spheres: &[Sphere; 4], ro: V3, rd: V3) -> V3 {
    let Some((t, hit)) = trace(spheres, ro, rd) else {
        return sky(rd);
    };

    let p = ro.add(rd.scale(t));
    let n = normal_at(spheres, &hit, p);
    let direct = shade(spheres, &hit, p, n);

    let mirror = mirror_of(spheres, &hit);
    if mirror <= 0.01 {
        return direct;
    }

    // One bounce only. A second would double the cost for a difference nobody
    // sees at this resolution.
    let rdir = rd.reflect(n);
    let rorig = p.add(n.scale(EPS * 8.0));
    let reflected = match trace(spheres, rorig, rdir) {
        Some((rt, rhit)) => {
            let rp = rorig.add(rdir.scale(rt));
            let rn = normal_at(spheres, &rhit, rp);
            shade(spheres, &rhit, rp, rn)
        }
        None => sky(rdir),
    };

    direct.scale(1.0 - mirror).add(reflected.scale(mirror))
}

// ---------------------------------------------------------------- pixels

/// Pack a colour into the PSP's 8888 pixel format, which stores red in the
/// lowest byte. Values above 1.0 are clipped rather than scaled.
fn pack(c: V3) -> u32 {
    let to_byte = |v: f32| -> u32 {
        let v = if v <= 0.0 {
            0.0
        } else if v >= 1.0 {
            1.0
        } else {
            // Gamma, roughly. Without it the shadowed halves look like holes.
            libm::sqrtf(v)
        };
        (v * 255.0) as u32
    };
    0xff00_0000 | (to_byte(c.z) << 16) | (to_byte(c.y) << 8) | to_byte(c.x)
}

// ------------------------------------------------------------ result file

/// Writes `value` into `buf` as decimal, returning how many bytes were used.
fn write_u64(buf: &mut [u8], mut value: u64) -> usize {
    if value == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut digits = [0u8; 20];
    let mut n = 0;
    while value > 0 {
        digits[n] = b'0' + (value % 10) as u8;
        value /= 10;
        n += 1;
    }
    for i in 0..n {
        buf[i] = digits[n - 1 - i];
    }
    n
}

/// Appends `s`, returning the new write position.
fn put(buf: &mut [u8], at: usize, s: &[u8]) -> usize {
    buf[at..at + s.len()].copy_from_slice(s);
    at + s.len()
}

/// Appends a number with three decimal places, given its value times 1000.
fn put_milli(buf: &mut [u8], mut at: usize, milli: u64) -> usize {
    at += write_u64(&mut buf[at..], milli / 1000);
    at = put(buf, at, b".");
    let frac = milli % 1000;
    if frac < 100 {
        at = put(buf, at, b"0");
    }
    if frac < 10 {
        at = put(buf, at, b"0");
    }
    at + write_u64(&mut buf[at..], frac)
}

/// Writes the run's numbers where psp-devloop can find them. The frame rate is
/// carried as thousandths so that no float formatting is needed.
fn write_result(frames: u32, micros: u64) {
    let mut buf = [0u8; 320];
    let mut at = 0;

    let fps_milli = if micros == 0 {
        0
    } else {
        frames as u64 * 1_000_000_000 / micros
    };
    let frame_ms_milli = if frames == 0 {
        0
    } else {
        micros * 1000 / frames as u64 / 1000
    };

    at = put(&mut buf, at, b"{\"demo\":\"psp-raytracer\",\"frames\":");
    at += write_u64(&mut buf[at..], frames as u64);
    at = put(&mut buf, at, b",\"elapsed_us\":");
    at += write_u64(&mut buf[at..], micros);
    at = put(&mut buf, at, b",\"fps\":");
    at = put_milli(&mut buf, at, fps_milli);
    at = put(&mut buf, at, b",\"frame_ms\":");
    at = put_milli(&mut buf, at, frame_ms_milli);
    at = put(&mut buf, at, b",\"render_width\":");
    at += write_u64(&mut buf[at..], RENDER_W as u64);
    at = put(&mut buf, at, b",\"render_height\":");
    at += write_u64(&mut buf[at..], RENDER_H as u64);
    at = put(&mut buf, at, b",\"scale\":");
    at += write_u64(&mut buf[at..], SCALE as u64);
    at = put(&mut buf, at, b"}\n");

    unsafe {
        let fd = sys::sceIoOpen(
            b"ms0:/raytracer-result.json\0".as_ptr(),
            sys::IoOpenFlags::WR_ONLY | sys::IoOpenFlags::CREAT | sys::IoOpenFlags::TRUNC,
            0o777,
        );
        if fd.0 >= 0 {
            sys::sceIoWrite(fd, buf.as_ptr() as *const _, at);
            sys::sceIoClose(fd);
        }
    }
}

// -------------------------------------------------------------------- main

fn psp_main() {
    psp::enable_home_button();

    unsafe {
        sys::sceDisplaySetMode(
            sys::DisplayMode::Lcd,
            SCREEN_WIDTH as usize,
            SCREEN_HEIGHT as usize,
        );

        // Cache-through address, so writes reach the display without an
        // explicit cache flush.
        let vram = (0x4000_0000u32 | sys::sceGeEdramGetAddr() as u32) as *mut u32;

        sys::sceDisplaySetFrameBuf(
            vram as *const u8,
            BUF_WIDTH as usize,
            sys::DisplayPixelFormat::Psm8888,
            sys::DisplaySetBufSync::NextFrame,
        );

        let tick_hz = sys::sceRtcGetTickResolution() as u64;
        let mut start: u64 = 0;
        sys::sceRtcGetCurrentTick(&mut start);

        let aspect = RENDER_W as f32 / RENDER_H as f32;

        for frame in 0..FRAMES {
            let t = frame as f32 * (1.0 / 60.0);
            let spheres = scene(t);

            // The camera circles the scene slowly, so the reflections move.
            let ca = t * 0.25;
            let eye = v3(libm::sinf(ca) * 1.6, 1.35, -4.6 + libm::cosf(ca) * 0.5);
            let target = v3(0.0, 0.0, 0.5);

            let fwd = target.sub(eye).norm();
            let right = v3(fwd.z, 0.0, -fwd.x).norm();
            let up = fwd.cross(right);

            for py in 0..RENDER_H {
                // Screen space, +y up.
                let sy = 1.0 - 2.0 * (py as f32 + 0.5) / RENDER_H as f32;

                for px in 0..RENDER_W {
                    let sx = (2.0 * (px as f32 + 0.5) / RENDER_W as f32 - 1.0) * aspect;

                    let dir = fwd
                        .add(right.scale(sx * 0.62))
                        .add(up.scale(sy * 0.62))
                        .norm();

                    let color = pack(radiance(&spheres, eye, dir));

                    // Blow the traced pixel up into a SCALE x SCALE block.
                    let base = py * SCALE * BUF_WIDTH as usize + px * SCALE;
                    for row in 0..SCALE {
                        let line = vram.add(base + row * BUF_WIDTH as usize);
                        for col in 0..SCALE {
                            *line.add(col) = color;
                        }
                    }
                }
            }
        }

        let mut end: u64 = 0;
        sys::sceRtcGetCurrentTick(&mut end);
        let micros = (end - start) * 1_000_000 / tick_hz;

        write_result(FRAMES, micros);

        sys::sceKernelExitGame();
    }
}
