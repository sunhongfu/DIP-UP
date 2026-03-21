"""
DIP-UP – Command-line interface
Usage:
    python run.py --phase ph.nii.gz --mask mask.nii.gz --output ./results/
    python run.py --help
"""

import argparse

from inference import run_dipup


def main():
    parser = argparse.ArgumentParser(
        description="DIP-UP: Deep Image Prior phase unwrapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase",      required=True, metavar="FILE",
                        help="Wrapped phase NIfTI (.nii / .nii.gz), 3D, values in radians.")
    parser.add_argument("--mask",       metavar="FILE", default=None,
                        help="Brain mask NIfTI (optional; non-zero voxels used if omitted).")
    parser.add_argument("--checkpoint", metavar="FILE", default=None,
                        help="Path to PhaseNet3D .pth checkpoint (random init if omitted).")
    parser.add_argument("--output",     metavar="DIR",  default="./dipup_output",
                        help="Output directory.")
    parser.add_argument("--n-iter",     type=int, default=2000, metavar="N",
                        help="Number of optimisation iterations.")
    parser.add_argument("--lr",         type=float, default=1e-6,
                        help="RMSprop learning rate.")
    parser.add_argument("--shift-base", type=int, default=5,
                        help="Winding number shift base.")

    args = parser.parse_args()

    unwrapped_path = run_dipup(
        phase_nii_path=args.phase,
        mask_nii_path=args.mask,
        checkpoint_path=args.checkpoint,
        n_iter=args.n_iter,
        lr=args.lr,
        shift_base=args.shift_base,
        output_dir=args.output,
    )

    print(f"\nOutputs:")
    print(f"  Unwrapped phase: {unwrapped_path}")


if __name__ == "__main__":
    main()
