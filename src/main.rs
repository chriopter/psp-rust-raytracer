//! A real-time ray tracer for the Sony PSP.
//!
//! Four spheres orbiting over an infinite checkerboard, with one reflection
//! bounce and hard shadows, traced per pixel on a 333 MHz MIPS Allegrex. The
//! PSP's GPU cannot do any of this: it is a fixed-function unit that textures
//! triangles, so every ray here is followed in software on the CPU.
//!
//! Every pixel of the 480x272 panel is shaded every frame, at full resolution
//! and with no temporal trickery — but only the pixels inside the spheres'
//! screen-space rectangle are ray traced. The rest is background, which is
//! most of the screen and a fraction of the cost.
//!
//! After `FRAMES` frames it writes ms0:/raytracer-result.json and exits, so
//! that a script can decide whether the run passed instead of a person
//! having to watch the screen.

#![no_std]
#![no_main]
// Inline assembly for MIPS is still unstable, and the square root below needs
// it. Nothing else here depends on nightly.
#![feature(asm_experimental_arch)]

use psp::sys;
use psp::{Align16, BUF_WIDTH, SCREEN_HEIGHT, SCREEN_WIDTH};

psp::module!("psp_raytracer", 1, 0);

/// Every pixel of the panel is its own ray — but not in the same frame.
///
/// Each frame traces one position out of every 2x2 block and leaves the other
/// three standing from earlier frames, so a quarter of the screen is refreshed
/// per frame and every pixel is renewed every four. This is the trick consoles
/// call checkerboard rendering: the picture is native, the cost is a quarter of
/// native, and the price is a smear behind fast movement. The order below
/// spreads the four positions apart in time rather than sweeping across.
/// How many pixels one frame writes, for the result file. Every one of them is
/// shaded; only those inside the spheres' screen rectangle are ray traced.
const RAYS_PER_FRAME: usize = SCREEN_WIDTH as usize * SCREEN_HEIGHT as usize;

/// How far the camera stands from the scene as it circles it.
const ORBIT_RADIUS: f32 = 4.8;

/// Frames to render in benchmark mode before writing the result and exiting.
///
/// Without ms0:/benchmark the demo runs until the PSP is switched off, which is
/// what you want when looking at it; with it, a caller gets a run that ends
/// by itself and a file with numbers in it.
const FRAMES: u32 = 300;

/// How far a ray may travel before it is considered to have hit nothing.
const FAR: f32 = 1.0e9;
/// Offset along the normal when spawning secondary rays, so that a ray does
/// not immediately hit the surface it started from.
const EPS: f32 = 1.0e-3;

/// Square root through the Allegrex FPU's own instruction.
///
/// `libm::sqrtf` is a software implementation in portable Rust. Ray tracing is
/// unusually square-root heavy — every normalisation and every sphere
/// intersection needs one — so it is worth spending one instruction instead.
#[inline(always)]
fn sqrtf(x: f32) -> f32 {
    let out: f32;
    unsafe {
        core::arch::asm!(
            "sqrt.s {out}, {x}",
            x = in(freg) x,
            out = out(freg) out,
            options(nostack, nomem, pure),
        );
    }
    out
}

/// Reciprocal square root without a square root or a division.
///
/// Both of those are among the slowest things the Allegrex FPU does, and
/// normalising a vector needs one of each. The bit pattern of a float is close
/// enough to its own logarithm that halving and subtracting from a constant
/// lands within a few percent of the answer; two Newton steps take it to
/// single-precision accuracy. The constant is Quake III's, which is the right
/// vintage for the hardware.
#[inline(always)]
fn rsqrtf(x: f32) -> f32 {
    let mut y = f32::from_bits(0x5f37_59df - (x.to_bits() >> 1));
    let half = x * 0.5;
    y *= 1.5 - half * y * y;
    y *= 1.5 - half * y * y;
    y
}

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
        self.scale(rsqrtf(len2))
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
    /// Kept alongside the radius so the surface normal needs a multiply rather
    /// than a normalisation: a hit point lies on the sphere by construction, so
    /// its distance from the centre is already known exactly.
    inv_radius: f32,
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
            inv_radius: 1.0 / 1.05,
            albedo: v3(0.92, 0.94, 0.97),
            mirror: 0.85,
        },
        Sphere {
            center: orbit(0.0, 2.1, 0.9, 1.7),
            radius: 0.5,
            inv_radius: 1.0 / 0.5,
            albedo: v3(0.90, 0.24, 0.18),
            mirror: 0.25,
        },
        Sphere {
            center: orbit(2.094, 2.1, 0.9, 1.3),
            radius: 0.5,
            inv_radius: 1.0 / 0.5,
            albedo: v3(0.20, 0.62, 0.90),
            mirror: 0.25,
        },
        Sphere {
            center: orbit(4.188, 2.1, 0.9, 2.1),
            radius: 0.5,
            inv_radius: 1.0 / 0.5,
            albedo: v3(0.96, 0.78, 0.16),
            mirror: 0.25,
        },
    ]
}

/// A sphere that contains every sphere in the scene at every point of the
/// animation. Most of the screen is sky or distant floor, where a ray cannot
/// reach any of them, and one test against this bound rejects all four at once
/// instead of intersecting each in turn.
const BOUND_CENTER: V3 = v3(0.0, -0.1, 0.5);
const BOUND_RADIUS: f32 = 3.3;

/// Whether a ray can reach the scene's spheres at all.
#[inline(always)]
fn misses_bound(ro: V3, rd: V3) -> bool {
    let oc = ro.sub(BOUND_CENTER);
    let b = oc.dot(rd);
    let c = oc.dot(oc) - BOUND_RADIUS * BOUND_RADIUS;
    // Pointing away with the origin outside, or missing the bound entirely.
    (b > 0.0 && c > 0.0) || b * b - c < 0.0
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
    let root = sqrtf(disc);
    let near = -b - root;
    if near > EPS {
        return Some(near);
    }
    let far = -b + root;
    if far > EPS { Some(far) } else { None }
}

/// Nearest intersection with the whole scene.
fn trace(spheres: &[Sphere; 4], ro: V3, rd: V3) -> Option<(f32, Hit)> {
    trace_with(spheres, ro, rd, true, true)
}

/// `check_bound` is worth its cost for rays that mostly miss — shadow and
/// reflection rays — and is pure overhead for a primary ray, which is only
/// ever cast at a pixel inside a silhouette and therefore always passes.
fn trace_with(
    spheres: &[Sphere; 4],
    ro: V3,
    rd: V3,
    check_bound: bool,
    check_floor: bool,
) -> Option<(f32, Hit)> {
    let mut best = FAR;
    let mut what = None;

    if !(check_bound && misses_bound(ro, rd)) {
        for (i, s) in spheres.iter().enumerate() {
            if let Some(t) = hit_sphere(s, ro, rd)
                && t < best
            {
                best = t;
                what = Some(Hit::Sphere(i));
            }
        }
    }

    // The floor is an infinite plane, so it needs no geometry at all: one
    // division says where the ray crosses y = FLOOR_Y. A primary ray skips it:
    // a floor hit there is discarded anyway, since the GE has already drawn the
    // floor, and the division is not cheap.
    if check_floor && rd.y < -1.0e-4 {
        let t = (FLOOR_Y - ro.y) / rd.y;
        if t > EPS && t < best {
            best = t;
            what = Some(Hit::Floor);
        }
    }

    what.map(|w| (best, w))
}

/// True when something stands between `p` and the light.
fn in_shadow(spheres: &[Sphere; 4], p: V3, self_index: Option<usize>) -> bool {
    let to_light = LIGHT.sub(p);
    let len2 = to_light.dot(to_light);
    let inv = rsqrtf(len2);
    let dist = len2 * inv; // len2 / sqrt(len2), without the division
    let dir = to_light.scale(inv);

    if misses_bound(p, dir) {
        return false;
    }

    spheres
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != self_index)
        .any(|(_, s)| matches!(hit_sphere(s, p, dir), Some(t) if t < dist))
}

/// Colour of the sky in a given direction: a plain vertical gradient, which
/// also gives the mirrored sphere something to reflect.
fn sky(rd: V3) -> V3 {
    let t = (rd.y * 0.5 + 0.5).clamp(0.0, 1.0);
    v3(0.06, 0.09, 0.16).scale(1.0 - t).add(v3(0.35, 0.52, 0.85).scale(t))
}

/// Shade one hit point, without following any further rays.
fn shade(spheres: &[Sphere; 4], hit: &Hit, p: V3, n: V3, occlude: bool) -> V3 {
    let albedo = match hit {
        Hit::Sphere(i) => spheres[*i].albedo,
        Hit::Floor => {
            // Checkerboard: the parity of the two floored coordinates.
            // Truncation plus a correction for negatives is a couple of
            // instructions, where libm's floor is a function call — and the
            // chrome sphere's reflection lands here constantly.
            let cx = p.x as i32 - (p.x < 0.0) as i32;
            let cz = p.z as i32 - (p.z < 0.0) as i32;
            if (cx + cz) & 1 == 0 {
                v3(0.83, 0.83, 0.86)
            } else {
                v3(0.13, 0.14, 0.17)
            }
        }
    };

    let to_light = LIGHT.sub(p).norm();
    let mut light = 0.16; // ambient, so shadows are dark but not black

    // A surface tilted away from the light is dark no matter what stands in
    // the way, so the shadow ray is only worth firing when it can change the
    // answer. On a scene this open that skips it for most of the sphere.
    let diffuse = n.dot(to_light);
    let self_index = match hit {
        Hit::Sphere(i) => Some(*i),
        Hit::Floor => None,
    };
    if diffuse > 0.0
        && !(occlude && in_shadow(spheres, p.add(n.scale(EPS * 8.0)), self_index))
    {
        light += diffuse * 0.9;

        // A tight highlight, which is what makes them read as balls rather
        // than flat discs. Raising the same cosine to the eighth power is
        // cheaper than a real half-vector and looks the same here.
        let d2 = diffuse * diffuse;
        let d4 = d2 * d2;
        light += d4 * d4 * 0.35;
    }

    albedo.scale(light)
}

/// Surface normal at a hit point.
fn normal_at(spheres: &[Sphere; 4], hit: &Hit, p: V3) -> V3 {
    match hit {
        Hit::Sphere(i) => p.sub(spheres[*i].center).scale(spheres[*i].inv_radius),
        Hit::Floor => v3(0.0, 1.0, 0.0),
    }
}

/// Where the floor stops reflecting. Most of the screen is distant floor, and
/// each reflecting pixel costs a second ray plus its shadow ray, so cutting it
/// off is the single largest saving available. Fading rather than cutting
/// avoids a visible ring across the checkerboard.
const REFLECT_NEAR: f32 = 8.0;
const REFLECT_FAR: f32 = 14.0;

/// How mirrored the surface is, at a hit `dist` away.
fn mirror_of(spheres: &[Sphere; 4], hit: &Hit, dist: f32) -> f32 {
    match hit {
        Hit::Sphere(i) => spheres[*i].mirror,
        // A wet-looking floor; it also proves the reflection ray works, because
        // the spheres appear upside down in it.
        Hit::Floor => {
            if dist >= REFLECT_FAR {
                0.0
            } else if dist <= REFLECT_NEAR {
                0.22
            } else {
                0.22 * (REFLECT_FAR - dist) / (REFLECT_FAR - REFLECT_NEAR)
            }
        }
    }
}

/// Follow one primary ray, and one bounce where the surface is mirrored.
///
/// Returns `None` when the ray reaches no sphere. That pixel belongs to the
/// floor or the sky, both of which the GE has already drawn, so leaving it
/// untouched is both correct and the cheapest possible answer — and it is why
/// the spheres' bounding rectangles may be generous.
fn sphere_radiance(spheres: &[Sphere; 4], ro: V3, rd: V3) -> Option<V3> {
    let (t, hit) = trace_with(spheres, ro, rd, false, false)?;

    let p = ro.add(rd.scale(t));
    let n = normal_at(spheres, &hit, p);
    let direct = shade(spheres, &hit, p, n, true);

    let mirror = mirror_of(spheres, &hit, t);
    if mirror <= 0.01 {
        return Some(direct);
    }

    // One bounce only. A second would double the cost for a difference nobody
    // sees at this size. The bounce may land on the floor, which is where the
    // checkerboard in the chrome sphere comes from.
    let rdir = rd.reflect(n);

    // A faintly mirrored surface only needs a sheen. Tracing the scene again
    // for a quarter-strength reflection costs as much as the primary ray and
    // shows almost nothing; the sky alone reads the same.
    if mirror < 0.5 {
        return Some(direct.scale(1.0 - mirror).add(sky(rdir).scale(mirror)));
    }

    let rorig = p.add(n.scale(EPS * 8.0));
    let reflected = match trace(spheres, rorig, rdir) {
        Some((rt, rhit)) => {
            let rp = rorig.add(rdir.scale(rt));
            let rn = normal_at(spheres, &rhit, rp);
            // No shadow ray on the bounce: a missing shadow inside a
            // reflection is not something anyone can point to, and it is a
            // quarter of the rays in the scene.
            shade(spheres, &rhit, rp, rn, false)
        }
        None => sky(rdir),
    };

    Some(direct.scale(1.0 - mirror).add(reflected.scale(mirror)))
}

// ------------------------------------------------------- screen-space bounds

/// A sphere's silhouette on screen: a circle, because the horizontal and
/// vertical scales work out equal for this projection.
///
/// Rectangles waste nearly half their pixels on the floor behind the sphere —
/// measured, not guessed — and each of those costs a full trace that ends up
/// discarded. Walking the circle instead skips them.
#[derive(Clone, Copy)]
struct Disc {
    cx: f32,
    cy: f32,
    /// Squared, because that is the form the per-pixel test needs.
    r2: f32,
    r: f32,
}

/// A rectangle of pixels, half-open on the far edge.
struct Rect {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// The camera basis, kept together because projecting needs all of it.
struct Cam {
    eye: V3,
    fwd: V3,
    right: V3,
    up: V3,
    k: f32,
    aspect: f32,
}

/// One pixel rectangle per sphere.
///
/// Rays are only worth casting where a sphere might be. Everything else is
/// floor or sky, which is background work: cheap here, and free once the GE
/// draws it. One rectangle per sphere rather than one around all four matters
/// enormously — four spheres spread across the scene have a union that covers
/// half the screen, while their own rectangles cover a small fraction of it.
///
/// Projecting the eight corners of each sphere's bounding cube is crude but
/// certain to enclose the sphere, and 32 projections per frame cost nothing
/// next to tens of thousands of rays.
fn sphere_bounds(
    cam: &Cam,
    spheres: &[Sphere; 4],
    w: usize,
    h: usize,
) -> ([Rect; 4], [Disc; 4]) {
    let mut discs = [Disc { cx: 0.0, cy: 0.0, r2: -1.0, r: 0.0 }; 4];
    let mut rects = [
        Rect { x0: 0, y0: 0, x1: 0, y1: 0 },
        Rect { x0: 0, y0: 0, x1: 0, y1: 0 },
        Rect { x0: 0, y0: 0, x1: 0, y1: 0 },
        Rect { x0: 0, y0: 0, x1: 0, y1: 0 },
    ];

    for (idx, s) in spheres.iter().enumerate() {
    let mut x0 = w as f32;
    let mut y0 = h as f32;
    let mut x1 = 0.0f32;
    let mut y1 = 0.0f32;

    {
        for corner in 0..8 {
            let sign = |bit: usize| if corner & (1 << bit) == 0 { -1.0 } else { 1.0 };
            let p = v3(
                s.center.x + sign(0) * s.radius,
                s.center.y + sign(1) * s.radius,
                s.center.z + sign(2) * s.radius,
            );

            let v = p.sub(cam.eye);
            let depth = v.dot(cam.fwd);
            if depth <= 0.05 {
                // Level with or behind the eye; fall back to the whole screen
                // for this sphere rather than projecting a wrong box.
                x0 = 0.0;
                y0 = 0.0;
                x1 = w as f32;
                y1 = h as f32;
                break;
            }

            let sx = v.dot(cam.right) / (depth * cam.k);
            let sy = v.dot(cam.up) / (depth * cam.k);

            let px = (sx / cam.aspect + 1.0) * 0.5 * w as f32;
            let py = (1.0 - sy) * 0.5 * h as f32;

            if px < x0 { x0 = px; }
            if px > x1 { x1 = px; }
            if py < y0 { y0 = py; }
            if py > y1 { y1 = py; }
        }
    }

    // A pixel of margin, because the corner projection encloses the cube
    // around the sphere rather than the sphere's curved silhouette.
    let clampi = |v: f32, lo: usize, hi: usize| -> usize {
        if v < lo as f32 { lo } else if v > hi as f32 { hi } else { v as usize }
    };
    rects[idx] = Rect {
        x0: clampi(x0 - 1.0, 0, w),
        y0: clampi(y0 - 1.0, 0, h),
        x1: clampi(x1 + 2.0, 0, w),
        y1: clampi(y1 + 2.0, 0, h),
    };

    // The silhouette itself, from the centre and the angular radius. A pixel
    // of margin covers the difference between this circle and the true conic
    // a sphere projects to when it sits off to one side.
    let v = s.center.sub(cam.eye);
    let depth = v.dot(cam.fwd);
    let dist2 = v.dot(v);
    if depth > s.radius + 0.05 && dist2 > s.radius * s.radius {
        let sx = v.dot(cam.right) / (depth * cam.k);
        let sy = v.dot(cam.up) / (depth * cam.k);
        // tan of the angular radius, in the same units as sx and sy.
        let sin_a = s.radius * rsqrtf(dist2);
        let cos_a = sqrtf(1.0 - sin_a * sin_a);
        let r_units = sin_a / (cos_a * cam.k);
        // Vertical and horizontal pixel scales are equal here, because the
        // aspect ratio cancels: w / (2 * aspect) == h / 2.
        let r_px = r_units * h as f32 * 0.5 + 1.0;
        discs[idx] = Disc {
            cx: (sx / cam.aspect + 1.0) * 0.5 * w as f32,
            cy: (1.0 - sy) * 0.5 * h as f32,
            r2: r_px * r_px,
            r: r_px,
        };
    } else {
        // Too close or behind: fall back to the rectangle by making the disc
        // cover it entirely.
        discs[idx] = Disc {
            cx: w as f32 * 0.5,
            cy: h as f32 * 0.5,
            r2: (w * w) as f32,
            r: w as f32,
        };
    }
    }

    (rects, discs)
}

// ------------------------------------------------------------- GE background

/// The floor and the sky are drawn by the Graphics Engine, not traced.
///
/// This is the whole reason the demo reaches a usable frame rate. A perspective
/// checkerboard on a plane and a vertical gradient are exactly what a
/// fixed-function unit from 2004 does for free, while tracing them was costing
/// three quarters of every frame: a floor pixel needs a shadow ray and a
/// reflection ray, which is as much work as a sphere.
///
/// The CPU then writes the ray-traced spheres straight into the same
/// framebuffer afterwards, so the two halves meet without a compositing pass.

/// Display list for the GE. Ordinary RAM; `sceGuStart` supplies the uncached
/// alias itself.
static mut LIST: Align16<[u32; 0x10000]> = Align16([0; 0x10000]);

/// Vertical field of view matching the ray generator, in degrees. The rays span
/// `atan(RAY_K)` above and below the centre, and the two pictures have to agree
/// exactly or the floor slides against the spheres standing on it.
const RAY_K: f32 = 0.62;

/// Half the world size of one checker square, so one texture repeat covers a
/// 2x2 block and lines up with the traced floor's `floor(x) + floor(z)` parity.
const TILE_WORLD: f32 = 2.0;
/// How far the floor quad reaches. Beyond this there is only sky.
const FLOOR_EXTENT: f32 = 400.0;

/// The floor's two colours, already carrying their lighting and gamma so the
/// GE can simply replace rather than modulate.
const TEXEL_LIGHT: u32 = 0xff_da_da_da;
const TEXEL_DARK: u32 = 0xff_56_52_4e;

const fn checker_tile() -> Align16<[u32; 16 * 16]> {
    let mut t = [0u32; 16 * 16];
    let mut i = 0;
    while i < t.len() {
        let x = i % 16;
        let y = i / 16;
        t[i] = if (x < 8) == (y < 8) { TEXEL_LIGHT } else { TEXEL_DARK };
        i += 1;
    }
    Align16(t)
}

static FLOOR_TEX: Align16<[u32; 16 * 16]> = checker_tile();

/// Texture coordinates come before position; the hardware fixes that order.
#[repr(C, align(4))]
struct TexVertex {
    u: f32,
    v: f32,
    x: f32,
    y: f32,
    z: f32,
}

const E: f32 = FLOOR_EXTENT;
const T: f32 = FLOOR_EXTENT / TILE_WORLD;

static FLOOR_QUAD: Align16<[TexVertex; 6]> = Align16([
    TexVertex { u: -T, v: -T, x: -E, y: FLOOR_Y, z: -E },
    TexVertex { u: -T, v:  T, x: -E, y: FLOOR_Y, z:  E },
    TexVertex { u:  T, v:  T, x:  E, y: FLOOR_Y, z:  E },
    TexVertex { u: -T, v: -T, x: -E, y: FLOOR_Y, z: -E },
    TexVertex { u:  T, v:  T, x:  E, y: FLOOR_Y, z:  E },
    TexVertex { u:  T, v: -T, x:  E, y: FLOOR_Y, z: -E },
]);

/// Screen-space vertices for the sky: colour first, then 16-bit position, and
/// padded to the four-byte stride the hardware reads.
#[repr(C, align(4))]
struct ColorVertex2D {
    color: u32,
    x: u16,
    y: u16,
    z: u16,
    _pad: u16,
}

/// The gradient the traced `sky()` produces, sampled at the top of the screen
/// and at the horizon, with gamma applied as `pack` would.
const SKY_TOP: u32 = 0xff_d3_a5_87;
const SKY_HORIZON: u32 = 0xff_b5_8d_73;

const W16: u16 = SCREEN_WIDTH as u16;
const H16: u16 = SCREEN_HEIGHT as u16;

static SKY_QUAD: Align16<[ColorVertex2D; 6]> = Align16([
    ColorVertex2D { color: SKY_TOP,     x: 0,   y: 0,   z: 0, _pad: 0 },
    ColorVertex2D { color: SKY_HORIZON, x: 0,   y: H16, z: 0, _pad: 0 },
    ColorVertex2D { color: SKY_HORIZON, x: W16, y: H16, z: 0, _pad: 0 },
    ColorVertex2D { color: SKY_TOP,     x: 0,   y: 0,   z: 0, _pad: 0 },
    ColorVertex2D { color: SKY_HORIZON, x: W16, y: H16, z: 0, _pad: 0 },
    ColorVertex2D { color: SKY_TOP,     x: W16, y: 0,   z: 0, _pad: 0 },
]);

/// A view matrix, built by hand.
///
/// `sceGumLookAt` cannot be used: its helper shadows the out-parameter with a
/// local of the same name, so the computed matrix is discarded and the caller
/// gets an identity. Upstream bug, not a misuse.
fn look_at(cam: &Cam) -> sys::ScePspFMatrix4 {
    let f = cam.fwd;
    let s = cam.right;
    let u = cam.up;
    let e = cam.eye;

    sys::ScePspFMatrix4 {
        x: sys::ScePspFVector4 { x: s.x, y: u.x, z: -f.x, w: 0.0 },
        y: sys::ScePspFVector4 { x: s.y, y: u.y, z: -f.y, w: 0.0 },
        z: sys::ScePspFVector4 { x: s.z, y: u.z, z: -f.z, w: 0.0 },
        w: sys::ScePspFVector4 {
            x: -s.dot(e),
            y: -u.dot(e),
            z: f.dot(e),
            w: 1.0,
        },
    }
}

/// One-time GE setup. Draw and display buffer are the same allocation and
/// `sceGuSwapBuffers` is never called, so the CPU can write into the buffer the
/// GE just drew.
unsafe fn gu_init(fb_offset: *mut core::ffi::c_void, depth_offset: *mut core::ffi::c_void) {
    unsafe {
        sys::sceGuInit();
        sys::sceGuStart(sys::GuContextType::Direct, &raw mut LIST as *mut core::ffi::c_void);
        sys::sceGuDrawBuffer(sys::DisplayPixelFormat::Psm8888, fb_offset, BUF_WIDTH as i32);
        sys::sceGuDispBuffer(
            SCREEN_WIDTH as i32,
            SCREEN_HEIGHT as i32,
            fb_offset,
            BUF_WIDTH as i32,
        );
        sys::sceGuDepthBuffer(depth_offset, BUF_WIDTH as i32);
        sys::sceGuOffset(2048 - SCREEN_WIDTH / 2, 2048 - SCREEN_HEIGHT / 2);
        sys::sceGuViewport(2048, 2048, SCREEN_WIDTH as i32, SCREEN_HEIGHT as i32);
        sys::sceGuScissor(0, 0, SCREEN_WIDTH as i32, SCREEN_HEIGHT as i32);
        sys::sceGuEnable(sys::GuState::ScissorTest);

        // Nothing here needs a depth test: the sky is painted first, the floor
        // over it, and the CPU's spheres last.
        sys::sceGuDisable(sys::GuState::DepthTest);
        sys::sceGuDepthMask(1);

        sys::sceGuFrontFace(sys::FrontFaceDirection::Clockwise);
        sys::sceGuDisable(sys::GuState::CullFace);
        sys::sceGuShadeModel(sys::ShadingModel::Smooth);
        sys::sceGuEnable(sys::GuState::ClipPlanes);

        sys::sceGuFinish();
        sys::sceGuSync(sys::GuSyncMode::Finish, sys::GuSyncBehavior::Wait);
        sys::sceDisplayWaitVblankStart();
        sys::sceGuDisplay(true);
    }
}

/// Queues sky and floor into the display list the caller has already started.
unsafe fn gu_background(cam: &Cam) {
    unsafe {
        // Sky: a screen-space gradient, no matrices involved.
        sys::sceGuDisable(sys::GuState::Texture2D);
        sys::sceGumDrawArray(
            sys::GuPrimitive::Triangles,
            sys::VertexType::COLOR_8888
                | sys::VertexType::VERTEX_16BIT
                | sys::VertexType::TRANSFORM_2D,
            6,
            core::ptr::null_mut(),
            &SKY_QUAD as *const _ as *const core::ffi::c_void,
        );

        // Floor: one enormous textured quad, tiled by wrapping.
        sys::sceGumMatrixMode(sys::MatrixMode::Projection);
        sys::sceGumLoadIdentity();
        sys::sceGumPerspective(
            2.0 * libm::atanf(RAY_K) * (180.0 / core::f32::consts::PI),
            cam.aspect,
            0.5,
            2000.0,
        );

        sys::sceGumMatrixMode(sys::MatrixMode::View);
        sys::sceGumLoadIdentity();
        let view = look_at(cam);
        sys::sceGumLoadMatrix(&view);

        sys::sceGumMatrixMode(sys::MatrixMode::Model);
        sys::sceGumLoadIdentity();

        sys::sceGuEnable(sys::GuState::Texture2D);
        sys::sceGuTexMode(sys::TexturePixelFormat::Psm8888, 0, 0, 0);
        sys::sceGuTexImage(
            sys::MipmapLevel::None,
            16,
            16,
            16,
            &FLOOR_TEX as *const _ as *const core::ffi::c_void,
        );
        sys::sceGuTexFunc(sys::TextureEffect::Replace, sys::TextureColorComponent::Rgb);
        sys::sceGuTexFilter(sys::TextureFilter::Linear, sys::TextureFilter::Linear);
        sys::sceGuTexWrap(sys::GuTexWrapMode::Repeat, sys::GuTexWrapMode::Repeat);
        sys::sceGuTexScale(1.0, 1.0);
        sys::sceGuTexOffset(0.0, 0.0);

        sys::sceGumDrawArray(
            sys::GuPrimitive::Triangles,
            sys::VertexType::TEXTURE_32BITF
                | sys::VertexType::VERTEX_32BITF
                | sys::VertexType::TRANSFORM_3D,
            6,
            core::ptr::null_mut(),
            &FLOOR_QUAD as *const _ as *const core::ffi::c_void,
        );
    }
}


// ------------------------------------------------------- sphere compositing

/// The spheres are traced into this buffer in main memory, not into the
/// framebuffer.
///
/// Writing rays straight into VRAM while the GE also renders there does not
/// work: the hardware and the emulators both treat a framebuffer the GE owns
/// as theirs, and stray CPU stores into it are lost. Handing the result to the
/// GE as a texture is the arrangement both understand.
///
/// Dimensions are powers of two because `sceGuTexImage` derives the size from
/// the leading zero count. Nothing clears it: every pixel inside a sphere's
/// rectangle is written each frame, opaque where a ray hit and transparent
/// where it missed, and nothing outside is ever sampled.
const SPRITE_STRIDE: usize = 512;
static mut SPRITE: Align16<[u32; SPRITE_STRIDE * 512]> = Align16([0; SPRITE_STRIDE * 512]);

/// Texture coordinates first, then a 16-bit screen position: the order the
/// hardware reads attributes in.
#[repr(C, align(4))]
struct SpriteVertex {
    u: f32,
    v: f32,
    x: u16,
    y: u16,
    z: u16,
    _pad: u16,
}

/// A bar in the corner saying which renderer is running: the two pictures are
/// close enough that without it you cannot always tell, which is rather the
/// point of the exercise.
static mut MODE_BAR: Align16<[ColorVertex2D; 2]> = Align16([
    ColorVertex2D { color: 0, x: 10, y: 10, z: 0, _pad: 0 },
    ColorVertex2D { color: 0, x: 118, y: 24, z: 0, _pad: 0 },
]);

const BAR_TRACED: u32 = 0xff_f0_c0_40;
const BAR_RASTER: u32 = 0xff_40_a0_f0;

unsafe fn gu_mode_bar(traced: bool) {
    unsafe {
        let color = if traced { BAR_TRACED } else { BAR_RASTER };
        let bar = &raw mut MODE_BAR;
        (*bar).0[0].color = color;
        (*bar).0[1].color = color;

        sys::sceGuDisable(sys::GuState::Texture2D);
        sys::sceGuDrawArray(
            sys::GuPrimitive::Sprites,
            sys::VertexType::COLOR_8888
                | sys::VertexType::VERTEX_16BIT
                | sys::VertexType::TRANSFORM_2D,
            2,
            core::ptr::null_mut(),
            (&raw const MODE_BAR) as *const core::ffi::c_void,
        );
    }
}

/// Two corners per sprite, four sprites.
static mut SPRITE_VERTS: Align16<[SpriteVertex; 8]> = Align16(
    [const {
        SpriteVertex { u: 0.0, v: 0.0, x: 0, y: 0, z: 0, _pad: 0 }
    }; 8],
);

/// Draws one finished frame: the GE's background, then the traced spheres over
/// it, in a single display list.
///
/// The alpha test throws away the pixels where a ray missed, so a generous
/// bounding rectangle costs nothing but a few discarded fragments.
unsafe fn gu_present(cam: &Cam, rects: &[Rect; 4], spheres: &[Sphere; 4], traced: bool) {
    unsafe {
        let mut count = 0usize;
        for r in rects {
            if r.x1 <= r.x0 || r.y1 <= r.y0 {
                continue;
            }
            let verts = &raw mut SPRITE_VERTS;
            (*verts).0[count * 2] = SpriteVertex {
                u: r.x0 as f32,
                v: r.y0 as f32,
                x: r.x0 as u16,
                y: r.y0 as u16,
                z: 0,
                _pad: 0,
            };
            (*verts).0[count * 2 + 1] = SpriteVertex {
                u: r.x1 as f32,
                v: r.y1 as f32,
                x: r.x1 as u16,
                y: r.y1 as u16,
                z: 0,
                _pad: 0,
            };
            count += 1;
        }

        // The GE reads main memory directly and never sees the data cache, so
        // everything written above has to be pushed out first.
        sys::sceKernelDcacheWritebackAll();

        // Background and spheres go out back to back, in one list, after the
        // tracing is finished. Drawing the background first and compositing a
        // tenth of a second later would leave the spheres on screen for only a
        // few milliseconds of every frame — which looks exactly like a
        // compositor that does not work.
        sys::sceGuStart(sys::GuContextType::Direct, &raw mut LIST as *mut core::ffi::c_void);
        gu_background(cam);

        if !traced {
            // Rasterised mode: the spheres are triangles, and nothing was
            // traced this frame.
            gu_spheres(spheres);
            gu_mode_bar(false);
            sys::sceGuFinish();
            sys::sceGuSync(sys::GuSyncMode::Finish, sys::GuSyncBehavior::Wait);
            return;
        }

        if count == 0 {
            gu_mode_bar(true);
            sys::sceGuFinish();
            sys::sceGuSync(sys::GuSyncMode::Finish, sys::GuSyncBehavior::Wait);
            return;
        }

        sys::sceGuEnable(sys::GuState::Texture2D);
        sys::sceGuTexMode(sys::TexturePixelFormat::Psm8888, 0, 0, 0);
        sys::sceGuTexImage(
            sys::MipmapLevel::None,
            SPRITE_STRIDE as i32,
            512,
            SPRITE_STRIDE as i32,
            &raw const SPRITE as *const core::ffi::c_void,
        );
        // Rgba, so the alpha written by the tracer survives to the test below.
        sys::sceGuTexFunc(sys::TextureEffect::Replace, sys::TextureColorComponent::Rgba);
        sys::sceGuTexFilter(sys::TextureFilter::Nearest, sys::TextureFilter::Nearest);
        sys::sceGuTexWrap(sys::GuTexWrapMode::Clamp, sys::GuTexWrapMode::Clamp);

        sys::sceGuEnable(sys::GuState::AlphaTest);
        sys::sceGuAlphaFunc(sys::AlphaFunc::Greater, 0, 0xff);

        sys::sceGuDrawArray(
            sys::GuPrimitive::Sprites,
            sys::VertexType::TEXTURE_32BITF
                | sys::VertexType::VERTEX_16BIT
                | sys::VertexType::TRANSFORM_2D,
            (count * 2) as i32,
            core::ptr::null_mut(),
            &raw const SPRITE_VERTS as *const core::ffi::c_void,
        );

        sys::sceGuDisable(sys::GuState::AlphaTest);
        gu_mode_bar(true);
        sys::sceGuFinish();
        sys::sceGuSync(sys::GuSyncMode::Finish, sys::GuSyncBehavior::Wait);
    }
}


// ------------------------------------------------------------------ matcap

/// The second renderer: the spheres are rasterised by the GE instead of traced.
///
/// The trick is where the shading comes from. At startup the ray tracer runs
/// once per material over every surface normal a sphere can present to the
/// camera, and stores the answer — light, highlight, reflection and all — in a
/// small texture indexed by that normal. A "matcap". From then on the GE draws
/// triangles and looks the answer up, which it does at sixty frames a second
/// without breaking a sweat.
///
/// Ray traced at load time, rasterised at run time. What it gives up is that
/// the reflection is fixed relative to the camera: as the camera circles, the
/// mirrored checkerboard stays put on the sphere instead of sliding across it.
/// Everything else survives.
const MATCAP: usize = 64;

static mut MATCAPS: [Align16<[u32; MATCAP * MATCAP]>; 4] =
    [const { Align16([0; MATCAP * MATCAP]) }; 4];

/// Fills one texture per sphere by tracing the sphere from every direction it
/// can face. Runs once.
unsafe fn bake_matcaps(spheres: &[Sphere; 4], cam: &Cam) {
    for idx in 0..4 {
        unsafe { bake_matcap(spheres, cam, idx) };
    }
}

/// One sphere's texture. Doing a single sphere per frame keeps the reflections
/// following the camera without the cost ever landing in one place: a whole
/// bake is about as much work as one traced frame, a quarter of it is not.
unsafe fn bake_matcap(spheres: &[Sphere; 4], cam: &Cam, only: usize) {
    unsafe {
        for (idx, sphere) in spheres.iter().enumerate() {
            if idx != only {
                continue;
            }
            let tex = (&raw mut MATCAPS[idx]) as *mut u32;

            for ty in 0..MATCAP {
                // Normal in view space: the texture is the sphere seen head on.
                let ny = 1.0 - 2.0 * (ty as f32 + 0.5) / MATCAP as f32;

                for tx in 0..MATCAP {
                    let nx = 2.0 * (tx as f32 + 0.5) / MATCAP as f32 - 1.0;
                    let r2 = nx * nx + ny * ny;
                    if r2 >= 1.0 {
                        // Outside the disc: never sampled, but leave it
                        // transparent rather than undefined.
                        *tex.add(ty * MATCAP + tx) = 0;
                        continue;
                    }
                    let nz = sqrtf(1.0 - r2);

                    // View space to world, through the camera's basis.
                    let n = cam
                        .right
                        .scale(nx)
                        .add(cam.up.scale(ny))
                        .add(cam.fwd.scale(-nz));

                    let p = sphere.center.add(n.scale(sphere.radius));
                    let hit = Hit::Sphere(idx);
                    let direct = shade(spheres, &hit, p, n, true);

                    // The same one bounce the traced renderer takes.
                    let view = n.scale(-1.0);
                    let mirror = sphere.mirror;
                    let color = if mirror <= 0.01 {
                        direct
                    } else {
                        let rdir = view.reflect(n);
                        let rorig = p.add(n.scale(EPS * 8.0));
                        let reflected = match trace(spheres, rorig, rdir) {
                            Some((rt, rhit)) => {
                                let rp = rorig.add(rdir.scale(rt));
                                let rn = normal_at(spheres, &rhit, rp);
                                shade(spheres, &rhit, rp, rn, false)
                            }
                            None => sky(rdir),
                        };
                        direct.scale(1.0 - mirror).add(reflected.scale(mirror))
                    };

                    *tex.add(ty * MATCAP + tx) = pack(color);
                }
            }
        }
    }
}

// -------------------------------------------------------------- sphere mesh

/// Rows and columns of the unit sphere the GE draws.
const MESH_LAT: usize = 12;
const MESH_LON: usize = 20;
const MESH_VERTS: usize = MESH_LAT * MESH_LON * 6;

static mut MESH: Align16<[TexVertex; MESH_VERTS]> = Align16(
    [const { TexVertex { u: 0.0, v: 0.0, x: 0.0, y: 0.0, z: 0.0 } }; MESH_VERTS],
);

/// Builds a unit sphere as a plain triangle list. Positions are also normals,
/// which is what makes the matcap lookup a two-line affair.
unsafe fn build_mesh() {
    unsafe {
        let mesh = (&raw mut MESH) as *mut TexVertex;
        let point = |lat: usize, lon: usize| -> V3 {
            let theta = lat as f32 * core::f32::consts::PI / MESH_LAT as f32;
            let phi = lon as f32 * 2.0 * core::f32::consts::PI / MESH_LON as f32;
            let st = libm::sinf(theta);
            v3(st * libm::cosf(phi), libm::cosf(theta), st * libm::sinf(phi))
        };

        let mut at = 0;
        for lat in 0..MESH_LAT {
            for lon in 0..MESH_LON {
                let a = point(lat, lon);
                let b = point(lat + 1, lon);
                let c = point(lat + 1, lon + 1);
                let d = point(lat, lon + 1);
                for p in [a, b, c, a, c, d] {
                    (*mesh.add(at)) = TexVertex { u: 0.0, v: 0.0, x: p.x, y: p.y, z: p.z };
                    at += 1;
                }
            }
        }
    }
}

/// Points every vertex's texture coordinate at the matcap texel for its normal.
///
/// The model transform is a translation and a uniform scale, so a vertex's
/// world normal is its position on the unit sphere — which means these
/// coordinates depend only on the camera and are shared by all four spheres.
unsafe fn update_mesh_uv(cam: &Cam) {
    unsafe {
        let mesh = (&raw mut MESH) as *mut TexVertex;
        for i in 0..MESH_VERTS {
            let vtx = &mut *mesh.add(i);
            let n = v3(vtx.x, vtx.y, vtx.z);
            vtx.u = (n.dot(cam.right) * 0.5 + 0.5) * MATCAP as f32;
            vtx.v = (0.5 - n.dot(cam.up) * 0.5) * MATCAP as f32;
        }
    }
}

/// Draws the four spheres as triangles, each with its own baked appearance.
unsafe fn gu_spheres(spheres: &[Sphere; 4]) {
    unsafe {
        sys::sceGuEnable(sys::GuState::Texture2D);
        sys::sceGuTexMode(sys::TexturePixelFormat::Psm8888, 0, 0, 0);
        sys::sceGuTexFunc(sys::TextureEffect::Replace, sys::TextureColorComponent::Rgb);
        sys::sceGuTexFilter(sys::TextureFilter::Linear, sys::TextureFilter::Linear);
        sys::sceGuTexWrap(sys::GuTexWrapMode::Clamp, sys::GuTexWrapMode::Clamp);
        sys::sceGuTexScale(1.0 / MATCAP as f32, 1.0 / MATCAP as f32);
        sys::sceGuTexOffset(0.0, 0.0);

        // Back faces would sample the matcap's far rim and show through.
        sys::sceGuFrontFace(sys::FrontFaceDirection::CounterClockwise);
        sys::sceGuEnable(sys::GuState::CullFace);

        for (idx, s) in spheres.iter().enumerate() {
            sys::sceGuTexImage(
                sys::MipmapLevel::None,
                MATCAP as i32,
                MATCAP as i32,
                MATCAP as i32,
                (&raw const MATCAPS[idx]) as *const core::ffi::c_void,
            );

            sys::sceGumMatrixMode(sys::MatrixMode::Model);
            sys::sceGumLoadIdentity();
            sys::sceGumTranslate(&sys::ScePspFVector3 {
                x: s.center.x,
                y: s.center.y,
                z: s.center.z,
            });
            sys::sceGumScale(&sys::ScePspFVector3 {
                x: s.radius,
                y: s.radius,
                z: s.radius,
            });

            sys::sceGumDrawArray(
                sys::GuPrimitive::Triangles,
                sys::VertexType::TEXTURE_32BITF
                    | sys::VertexType::VERTEX_32BITF
                    | sys::VertexType::TRANSFORM_3D,
                MESH_VERTS as i32,
                core::ptr::null_mut(),
                (&raw const MESH) as *const core::ffi::c_void,
            );
        }

        sys::sceGuDisable(sys::GuState::CullFace);
        sys::sceGumMatrixMode(sys::MatrixMode::Model);
        sys::sceGumLoadIdentity();
    }
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
            // Gamma, roughly; without it the shadowed halves look like holes.
            // Expressed as v * rsqrt(v), because the reciprocal square root is
            // a handful of multiplies while the FPU's square root is not, and
            // this runs three times for every pixel that gets traced.
            v * rsqrtf(v)
        };
        (v * 255.0) as u32
    };
    0xff00_0000 | (to_byte(c.z) << 16) | (to_byte(c.y) << 8) | to_byte(c.x)
}

/// Midpoint of two packed colours, channel by channel. Both are opaque, so the
/// alpha byte comes out unchanged.
#[inline(always)]
fn blend(a: u32, b: u32) -> u32 {
    // Halve both, then add: no channel can carry into its neighbour.
    ((a & 0xfefe_fefe) >> 1) + ((b & 0xfefe_fefe) >> 1) | 0xff00_0000
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

/// Writes the run's numbers where the caller can find them. The frame rate is
/// carried as thousandths so that no float formatting is needed.
fn write_result(
    frames: u32,
    micros: u64,
    work_micros: u64,
    traced_frames: u32,
    traced_micros: u64,
    raster_frames: u32,
    raster_micros: u64,
) {
    let mut buf = [0u8; 512];
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

    // The presented rate is quantised: waiting for vertical blank means a
    // frame lasts a whole number of 16.68 ms intervals, so only 60/n is
    // reachable. The work figure is what the frame actually costs, and is what
    // moves when the renderer gets faster.
    let work_ms_milli = if frames == 0 { 0 } else { work_micros / frames as u64 };
    at = put(&mut buf, at, b",\"work_ms\":");
    at = put_milli(&mut buf, at, work_ms_milli);
    at = put(&mut buf, at, b",\"work_fps\":");
    let work_fps_milli = if work_micros == 0 {
        0
    } else {
        frames as u64 * 1_000_000_000 / work_micros
    };
    at = put_milli(&mut buf, at, work_fps_milli);
    at = put(&mut buf, at, b",\"width\":");
    at += write_u64(&mut buf[at..], SCREEN_WIDTH as u64);
    at = put(&mut buf, at, b",\"height\":");
    at += write_u64(&mut buf[at..], SCREEN_HEIGHT as u64);
    at = put(&mut buf, at, b",\"rays_per_frame\":");
    at += write_u64(&mut buf[at..], RAYS_PER_FRAME as u64);
    // Each renderer's own cost, so a run that switched modes still says what
    // each one is worth.
    let mode = |buf: &mut [u8], mut at: usize, label: &[u8], f: u32, us: u64| -> usize {
        at = put(buf, at, label);
        at += write_u64(&mut buf[at..], f as u64);
        if f > 0 && us > 0 {
            at = put(buf, at, b",\"fps\":");
            at = put_milli(buf, at, f as u64 * 1_000_000_000 / us);
            at = put(buf, at, b",\"ms\":");
            at = put_milli(buf, at, us / f as u64);
        }
        put(buf, at, b"}")
    };
    at = mode(&mut buf, at, b",\"traced\":{\"frames\":", traced_frames, traced_micros);
    at = mode(&mut buf, at, b",\"rasterised\":{\"frames\":", raster_frames, raster_micros);

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
        // One framebuffer, shared: the GE draws the background into it and the
        // CPU writes spheres over that. Both halves must name the same memory,
        // which they do in different ways — the GE wants an offset from the
        // start of VRAM, the CPU an absolute cache-through address.
        let allocator = psp::vram_alloc::get_vram_allocator().unwrap();
        let fb = allocator.alloc_texture_pixels(
            BUF_WIDTH,
            SCREEN_HEIGHT,
            sys::TexturePixelFormat::Psm8888,
        );
        let depth = allocator.alloc_texture_pixels(
            BUF_WIDTH,
            SCREEN_HEIGHT,
            sys::TexturePixelFormat::Psm4444,
        );

        gu_init(
            fb.as_mut_ptr_from_zero() as *mut core::ffi::c_void,
            depth.as_mut_ptr_from_zero() as *mut core::ffi::c_void,
        );

        let tick_hz = sys::sceRtcGetTickResolution() as u64;
        let mut start: u64 = 0;
        sys::sceRtcGetCurrentTick(&mut start);

        let w = SCREEN_WIDTH as usize;
        let h = SCREEN_HEIGHT as usize;
        let aspect = w as f32 / h as f32;

        // Waiting here rather than inside the present keeps the wait out of
        // the work measurement below.
        let mut work_ticks: u64 = 0;

        // Both renderers start from the same scene. The matcaps are baked once
        // for the camera's opening position; see their definition for what that
        // costs in fidelity.
        sys::sceCtrlSetSamplingCycle(0);
        sys::sceCtrlSetSamplingMode(sys::CtrlMode::Digital);
        build_mesh();
        {
            let spheres0 = scene(0.0);
            let eye0 = v3(0.0, 1.35, -4.6 + 0.5);
            let fwd0 = v3(0.0, 0.0, 0.5).sub(eye0).norm();
            let right0 = v3(fwd0.z, 0.0, -fwd0.x).norm();
            let cam0 = Cam {
                eye: eye0,
                fwd: fwd0,
                right: right0,
                up: fwd0.cross(right0),
                k: RAY_K,
                aspect: SCREEN_WIDTH as f32 / SCREEN_HEIGHT as f32,
            };
            bake_matcaps(&spheres0, &cam0);
        }

        // A run with no benchmark request never ends, so the demo can just be
        // watched. L and R still switch renderers either way.
        let benchmark = {
            let fd = sys::sceIoOpen(
                b"ms0:/benchmark\0".as_ptr(),
                sys::IoOpenFlags::RD_ONLY,
                0o777,
            );
            if fd.0 >= 0 {
                sys::sceIoClose(fd);
                true
            } else {
                false
            }
        };
        let limit = if benchmark { FRAMES } else { u32::MAX };

        let mut traced = true;
        // Outside benchmark mode the demo alternates on its own every few
        // seconds, so it shows both renderers unattended. L and R override it.
        let switch_ticks = tick_hz * 6;
        let mut last_switch = start;
        let mut traced_frames: u32 = 0;
        let mut traced_ticks: u64 = 0;
        let mut raster_frames: u32 = 0;
        let mut raster_ticks: u64 = 0;

        for frame in 0..limit {
            sys::sceDisplayWaitVblankStart();

            // L picks the ray tracer, R the rasteriser.
            let mut pad = sys::SceCtrlData::default();
            sys::sceCtrlReadBufferPositive(&mut pad, 1);
            let mut now: u64 = 0;
            sys::sceRtcGetCurrentTick(&mut now);

            if pad.buttons.contains(sys::CtrlButtons::LTRIGGER) {
                traced = true;
                last_switch = now;
            } else if pad.buttons.contains(sys::CtrlButtons::RTRIGGER) {
                traced = false;
                last_switch = now;
            } else if !benchmark && now.wrapping_sub(last_switch) > switch_ticks {
                traced = !traced;
                last_switch = now;
            }

            let mut frame_start: u64 = 0;
            sys::sceRtcGetCurrentTick(&mut frame_start);

            let t = frame as f32 * (1.0 / 60.0);
            let spheres = scene(t);

            // A full orbit every twelve seconds. The movement is the point:
            // a still frame cannot show that the reflections in the chrome
            // sphere slide when they are traced and sit still when they are
            // looked up.
            let ca = t * 0.52;
            let eye = v3(
                libm::sinf(ca) * ORBIT_RADIUS,
                1.35,
                0.5 - libm::cosf(ca) * ORBIT_RADIUS,
            );
            let target = v3(0.0, 0.0, 0.5);

            let fwd = target.sub(eye).norm();
            let right = v3(fwd.z, 0.0, -fwd.x).norm();
            let up = fwd.cross(right);
            let cam = Cam { eye, fwd, right, up, k: RAY_K, aspect };



            // Rays are only cast where a sphere can be. Everything outside is
            // background, which costs a fraction as much because it needs
            // neither a shadow ray nor a reflection.
            update_mesh_uv(&cam);

            let (rects, discs) = sphere_bounds(&cam, &spheres, w, h);
            if !traced {
                // Refresh one sphere's baked appearance per frame, so the
                // reflections keep up with the orbit.
                bake_matcap(&spheres, &cam, (frame % 4) as usize);
                gu_present(&cam, &rects, &spheres, false);
                let mut frame_end: u64 = 0;
                sys::sceRtcGetCurrentTick(&mut frame_end);
                let d = frame_end - frame_start;
                work_ticks += d;
                raster_ticks += d;
                raster_frames += 1;
                continue;
            }

            let sprite = (&raw mut SPRITE) as *mut u32;

            // The GE composites whole rectangles, but only the silhouettes are
            // traced. Without clearing, the corners keep whatever a previous
            // frame left there and the spheres drag blocks around behind them.
            // These are plain stores; next to a traced pixel they are free.
            for r in &rects {
                for py in r.y0..r.y1 {
                    let row = sprite.add(py * SPRITE_STRIDE);
                    for px in r.x0..r.x1 {
                        *row.add(px) = 0;
                    }
                }
            }

            for py in 0..h {
                // Screen space, +y up.
                let sy = 1.0 - 2.0 * (py as f32 + 0.5) / h as f32;
                let fy = py as f32 + 0.5;

                // Where each silhouette crosses this scanline, as a half-open
                // pixel span. A circle contributes at most one.
                let mut spans: [(usize, usize); 4] = [(0, 0); 4];
                let mut n = 0;
                for d in &discs {
                    let dy = fy - d.cy;
                    let inside = d.r2 - dy * dy;
                    if inside <= 0.0 {
                        continue;
                    }
                    let half = sqrtf(inside);
                    let lo = d.cx - half;
                    let hi = d.cx + half;
                    if hi < 0.0 || lo >= w as f32 {
                        continue;
                    }
                    let lo = if lo < 0.0 { 0 } else { lo as usize };
                    let hi = if hi >= w as f32 { w } else { hi as usize + 1 };
                    if hi > lo {
                        spans[n] = (lo, hi);
                        n += 1;
                    }
                }
                if n == 0 {
                    continue;
                }

                // Merge overlaps, so a pixel two spheres share is traced once.
                // Insertion sort by start; four entries at most.
                for i in 1..n {
                    let mut j = i;
                    while j > 0 && spans[j - 1].0 > spans[j].0 {
                        spans.swap(j - 1, j);
                        j -= 1;
                    }
                }

                let row = sprite.add(py * SPRITE_STRIDE);
                let mut lo = spans[0].0;
                let mut hi = spans[0].1;

                for k in 1..=n {
                    if k < n && spans[k].0 <= hi {
                        if spans[k].1 > hi {
                            hi = spans[k].1;
                        }
                        continue;
                    }

                    // Trace every other pixel and interpolate between them.
                    // A sphere's shading varies smoothly, so the difference is
                    // invisible in the middle; at the silhouette the fill
                    // refuses to guess, which keeps the edge where it belongs.
                    let mut px = lo;
                    while px < hi {
                        let sx = (2.0 * (px as f32 + 0.5) / w as f32 - 1.0) * aspect;
                        let dir = fwd
                            .add(right.scale(sx * cam.k))
                            .add(up.scale(sy * cam.k))
                            .norm();

                        // Transparent where the ray reached no sphere: the
                        // silhouette is a hair wider than the sphere itself,
                        // and the GE's alpha test drops the difference.
                        *row.add(px) = match sphere_radiance(&spheres, eye, dir) {
                            Some(color) => pack(color),
                            None => 0,
                        };
                        px += 2;
                    }

                    // Make sure the span's last pixel is a traced one, so the
                    // fill below always has a right-hand neighbour.
                    let last = hi - 1;
                    if (last - lo) % 2 == 1 {
                        let sx = (2.0 * (last as f32 + 0.5) / w as f32 - 1.0) * aspect;
                        let dir = fwd
                            .add(right.scale(sx * cam.k))
                            .add(up.scale(sy * cam.k))
                            .norm();
                        *row.add(last) = match sphere_radiance(&spheres, eye, dir) {
                            Some(color) => pack(color),
                            None => 0,
                        };
                    }

                    let mut px = lo + 1;
                    while px < last {
                        let a = *row.add(px - 1);
                        let b = *row.add(px + 1);
                        *row.add(px) = if a == 0 || b == 0 {
                            // One side is off the sphere: this pixel sits on
                            // the silhouette, and averaging would smear it.
                            0
                        } else {
                            blend(a, b)
                        };
                        px += 2;
                    }

                    if k < n {
                        lo = spans[k].0;
                        hi = spans[k].1;
                    }
                }
            }

            gu_present(&cam, &rects, &spheres, true);

            let mut frame_end: u64 = 0;
            sys::sceRtcGetCurrentTick(&mut frame_end);
            let d = frame_end - frame_start;
            work_ticks += d;
            traced_ticks += d;
            traced_frames += 1;
        }

        let mut end: u64 = 0;
        sys::sceRtcGetCurrentTick(&mut end);
        let micros = (end - start) * 1_000_000 / tick_hz;
        let work_micros = work_ticks * 1_000_000 / tick_hz;

        write_result(
            FRAMES,
            micros,
            work_micros,
            traced_frames,
            traced_ticks * 1_000_000 / tick_hz,
            raster_frames,
            raster_ticks * 1_000_000 / tick_hz,
        );

        sys::sceKernelExitGame();
    }
}
