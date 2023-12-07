from csmap.process import CsmapParams, process


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dem_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--gf_size", type=int, default=12)
    parser.add_argument("--gf_sigma", type=int, default=3)
    parser.add_argument("--cvt_size", type=int, default=1)
    parser.add_argument("--height_scale", type=float, nargs=2, default=[0.0, 1000.0])
    parser.add_argument("--slope_scale", type=float, nargs=2, default=[0.0, 1.5])
    parser.add_argument("--curvature_scale", type=float, nargs=2, default=[-0.1, 0.1])

    args = parser.parse_args()

    params = CsmapParams(
        gf_size=args.gf_size,
        gf_sigma=args.gf_sigma,
        cvt_size=args.cvt_size,
        height_scale=args.height_scale,
        slope_scale=args.slope_scale,
        curvature_scale=args.curvature_scale,
    )

    return {
        "input_dem_path": args.input_dem_path,
        "output_path": args.output_path,
        "chunk_size": args.chunk_size,
        "params": params,
    }


def main():
    args = parse_args()
    process(**args)


if __name__ == "__main__":
    main()
