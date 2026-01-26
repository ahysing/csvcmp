#![feature(portable_simd)]
use clap::Parser;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::error::Error;
use std::path::Path;
use std::process::exit;
use std::simd::LaneCount;
use std::simd::Simd;
use std::simd::SupportedLaneCount;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::prelude::*;

struct Graph {
    x: Vec<f64>,
    y: Vec<f64>,
}

pub fn calculate_abs_area_simd<const LANES: usize>(
    ax: &[f64],
    ay: &[f64],
    bx: &[f64],
    by: &[f64],
) -> f64
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let x = merge_sorted(ax, bx);
    let mut total_area = 0.0;

    let mut i = 0;
    
    let mut idx_a_lane: Simd<usize, LANES> = Simd::splat(0);
    let mut idx_b_lane: Simd<usize, LANES> = Simd::splat(0);
    while i + LANES <= x.len() - 1 {
        idx_a_lane = scatter_max(idx_a_lane);
        idx_b_lane = scatter_max(idx_b_lane);

        let x0 = Simd::<f64, LANES>::from_slice(&x[i..i + LANES]);
        let x1 = Simd::<f64, LANES>::from_slice(&x[i + 1..i + 1 + LANES]);


        let ay0 = interpolate_simd::<LANES>(ax, ay, x0, idx_a_lane);
        let ay1 = interpolate_simd::<LANES>(ax, ay, x1, idx_b_lane);

        let by0 = interpolate_simd::<LANES>(bx, by, x0, idx_a_lane);
        let by1 = interpolate_simd::<LANES>(bx, by, x1, idx_b_lane);

        let y0 = ay0 - by0;
        let y1 = ay1 - by1;

        let sign0 = y0.signum();
        let sign1 = y1.signum();

        let same_mask = (sign0 * sign1).simd_gt(Simd::splat(0.0));
        let diff_mask = (sign0 * sign1).simd_lt(Simd::splat(0.0));
        let dx = (x1 - x0).to_array();
        let abs_y0 = y0.abs().to_array();
        let abs_y1 = y1.abs().to_array();
        for k in 0..LANES {
            unsafe {
                if same_mask.test_unchecked(k) {
                    let trapezoid_area = (abs_y0[k] + abs_y1[k]) * 0.5 * dx[k];
                    total_area += trapezoid_area;
                } else if diff_mask.test_unchecked(k) {
                    let t = abs_y0[k] / (abs_y0[k] + abs_y1[k]);
                    let dx1 = dx[k] * t;
                    let dx2 = dx[k] - dx1;
                    let tri_area = abs_y0[k] * dx1 * 0.5 + abs_y1[k] * dx2 * 0.5;
                    total_area += tri_area;
                }
            }
        }
        
        i += LANES;
    }

    let idx_a: usize = idx_a_lane.reduce_max();
    let idx_b: usize = idx_b_lane.reduce_max();
    for j in i..x.len() - 1 {
        let x0 = x[j];
        let x1 = x[j + 1];

        let y0 = interpolate(ax, ay, x0, idx_a) - interpolate(bx, by, x0, idx_b);
        let y1 = interpolate(ax, ay, x1, idx_a) - interpolate(bx, by, x1, idx_b);

        let dx = x1 - x0;

        if y0 == 0.0 && y1 == 0.0 {
            continue;
        }

        if y0.signum() == y1.signum() {
            total_area += (y0.abs() + y1.abs()) * 0.5 * dx;
        } else {
            let t = y0.abs() / (y0.abs() + y1.abs());
            let dx1 = dx * t;
            let dx2 = dx - dx1;

            total_area += y0.abs() * dx1 * 0.5;
            total_area += y1.abs() * dx2 * 0.5;
        }
    }

    total_area
}

fn merge_sorted(ax: &[f64], bx: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(ax.len() + bx.len());
    let (mut i, mut j) = (0, 0);

    while i < ax.len() && j < bx.len() {
        let a = ax[i];
        let b = bx[j];

        if a <= b {
            out.push(a);
            i += 1;
            if a == b {
                j += 1;
            }
        } else {
            out.push(b);
            j += 1;
        }
    }

    out.extend_from_slice(&ax[i..]);
    out.extend_from_slice(&bx[j..]);

    out
}

fn scatter_max<const LANES: usize>(
    values: Simd<usize, LANES>
) -> Simd<usize, LANES>
where
    LaneCount<LANES>: SupportedLaneCount {
    let max_val = values.reduce_max();
    Simd::<usize, LANES>::splat(max_val)
    
}

fn interpolate_simd<const LANES: usize>(
    xs: &[f64],
    ys: &[f64],
    a: Simd<f64, LANES>,
    mut idx: Simd<usize, LANES>,
) -> Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut out = Simd::splat(0.0);

    for lane in 0..LANES {
        let av = a[lane];

        let val = if av <= xs[0] {
            ys[0]
        } else if av >= xs[xs.len() - 1] {
            ys[ys.len() - 1]
        } else {
            let offset = xs[idx[lane]..].iter().position(|&p| p > av).unwrap();
            
            idx[lane] += offset;

            let i = idx[lane];
            let x0 = xs[i - 1];
            let y0 = ys[i - 1];
            let x1 = xs[i];
            let y1 = ys[i];

            y0 + (y1 - y0) * (av - x0) / (x1 - x0)
        };

        out[lane] = val;
    }

    out
}

fn interpolate(x: &[f64], y: &[f64], a: f64, mut i: usize) -> f64 {
    if a <= x[0] {
        return y[0];
    }
    if a >= x[x.len() - 1] {
        return y[y.len() - 1];
    }

    let offset = x[i..].iter().position(|&p| p > a).unwrap();
    i += offset;

    let x0 = x[i - 1];
    let y0 = y[i - 1];
    let x1 = x[i];
    let y1 = y[i];

    y0 + (y1 - y0) * (a - x0) / (x1 - x0)
}

#[derive(Parser, Debug)]
#[command(author, version, after_help = "Load two graphs from CSV files and compute the area between the graphs. The computation is the integral of absolute value of differnces.")]
struct Options {
    #[arg(short, long, help = "first CSV file")]
    first: String,

    #[arg(short, long, help = "second CSV file")]
    second: String,

    #[arg(short, long, default_value_t = 0, help = "x-axis column index (zero-based)")]
    x: usize,

    #[arg(short, long, default_value_t = 1, help = "y-axis column index (zero-based)")]
    y: usize,

    #[arg(long, help = "has header")]
    has_header: bool,
}

enum ExitCodeType {
    Ok = 0,
    FileNotFound = 1,
    NoData = 2,
    Failure = 3
}

fn main() {
    let opt = Options::parse();

    if !Path::new(&opt.first).exists() {
        eprintln!("CSV '{}' not found.", opt.first);
        exit(ExitCodeType::FileNotFound as i32);
    }
    if !Path::new(&opt.second).exists() {
        eprintln!("CSV '{}' not found.", opt.second);
        exit(ExitCodeType::FileNotFound as i32);
    }

    if opt.x == opt.y {
        eprintln!("Invalid column selected: {} and {}", opt.x, opt.y);
        exit(ExitCodeType::FileNotFound as i32);
    }

    let first = match read_csv(&opt.first, opt.x, opt.y, opt.has_header) {
        Ok(data) if !data.x.is_empty() => data,
        Ok(_) => {
            eprintln!("CSV '{}' no data found.", opt.first);
            exit(ExitCodeType::NoData as i32);
        }
        Err(e) => {
            eprintln!("{:?}", e);
            exit(ExitCodeType::Failure as i32);
        }
    };
    let second = match read_csv(&opt.second, opt.x, opt.y, opt.has_header) {
        Ok(data) if !data.x.is_empty() => data,
        Ok(_) => {
            eprintln!("CSV '{}' no data found.", opt.second);
            exit(ExitCodeType::NoData as i32);
        }
        Err(e) => {
            eprintln!("{:?}", e);
            exit(ExitCodeType::Failure as i32);
        }
    };
    let ax = first.x;
    let bx = second.x;
    let ay = first.y;
    let by = second.y;

    if ax.is_empty() {
        eprintln!("CSV '{}' column {} empty.", opt.first, opt.x);
        exit(ExitCodeType::NoData as i32);
    }
    if bx.is_empty() {
        eprintln!("CSV '{}' column {} empty.", opt.second, opt.x);
        exit(ExitCodeType::NoData as i32);
    }
    if ay.is_empty() {
        eprintln!("CSV '{}' column {} empty.", opt.first, opt.y);
        exit(ExitCodeType::NoData as i32);
    }
    if by.is_empty() {
        eprintln!("CSV '{}' column {} empty.", opt.second, opt.y);
        exit(ExitCodeType::NoData as i32);
    }
    if ax.len() != ay.len() {
        eprintln!("CSV '{}' length not matching.", opt.first);
        exit(ExitCodeType::Failure as i32);
    }
    if bx.len() != by.len() {
        eprintln!("CSV '{}' length not matching.", opt.second);
        exit(ExitCodeType::Failure as i32);
    }

    println!(
        "∫‖A − B‖ dx: {}",
        calculate_abs_area_simd::<8>(&ax, &ay, &bx, &by)
    );
    exit(ExitCodeType::Ok as i32);
}

fn read_csv(path: &str, x: usize, y: usize, has_header: bool)
    -> Result<Graph, Box<dyn Error>>
{
    // --- First pass: count lines ---
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Count non-empty lines
    let mut line_count = reader.lines().count();

    if has_header && line_count > 0 {
        line_count -= 1;
    }

    // --- Preallocate ---
    let mut xs = Vec::with_capacity(line_count);
    let mut ys = Vec::with_capacity(line_count);

    // --- Second pass: parse CSV ---
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(has_header)
        .from_path(path)?;

    for (line, result) in rdr.records().enumerate() {
        let record = result?;

        let sx = record.get(x)
            .ok_or_else(|| format!("Missing column {} at line {}", x, line))?
            .trim();

        let sy = record.get(y)
            .ok_or_else(|| format!("Missing column {} at line {}", y, line))?
            .trim();

        if sx.is_empty() || sy.is_empty() {
            return Err(format!("Empty numeric field at line {}", line).into());
        }

        let xv = sx.parse::<f64>()
            .map_err(|_| format!("Invalid float '{}' at line {}", sx, line))?;

        let yv = sy.parse::<f64>()
            .map_err(|_| format!("Invalid float '{}' at line {}", sy, line))?;

        xs.push(xv);
        ys.push(yv);
    }

    Ok(Graph { x: xs, y: ys })
}
