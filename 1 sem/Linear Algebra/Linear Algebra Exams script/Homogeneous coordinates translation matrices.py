from __future__ import annotations
import sys

try:
    import sympy as sp
except Exception as e:
    print("Could not import sympy. Install it first with: pip install sympy")
    print("Error:", e)
    sys.exit(1)


class InputError(Exception):
    pass


def parse_num(s: str) -> sp.Rational:
    s = s.strip()
    if s == "":
        raise InputError("Empty input.")
    try:
        return sp.Rational(s)  # supports -3, 1/2, 0.25
    except Exception:
        raise InputError(f"Could not parse '{s}'. Use e.g. 2, -3, 1/2, 0.5")


def read_int(prompt: str, min_value: int = 1, max_value: int = 50) -> int:
    while True:
        txt = input(prompt).strip()
        if txt.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not txt.isdigit():
            print("Please enter a positive integer.")
            continue
        v = int(txt)
        if v < min_value or v > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return v


def read_point(d: int, prompt: str) -> sp.Matrix:
    """
    Read a point in R^d as "x, y" or "x y" (or 3 numbers for 3D).
    Returns a column vector (d x 1).
    """
    while True:
        txt = input(prompt).strip()
        if txt.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        for ch in "()[]{}":
            txt = txt.replace(ch, "")
        txt = txt.replace(",", " ")
        parts = [p for p in txt.split() if p.strip() != ""]
        if len(parts) != d:
            print(f"❌ Please enter exactly {d} numbers (e.g. 1, -2{' , 3' if d==3 else ''}).")
            continue
        try:
            nums = [parse_num(p) for p in parts]
            return sp.Matrix(nums)
        except InputError as e:
            print(f"❌ {e}")


def to_homogeneous(p: sp.Matrix) -> sp.Matrix:
    # append 1
    return p.col_insert(1, sp.Matrix([1])) if p.cols == 1 else sp.Matrix(list(p) + [1])


def from_homogeneous(ph: sp.Matrix, d: int) -> sp.Matrix:
    # assume last entry is 1
    return ph[:d, :]


def translation_matrix_2d(tx, ty) -> sp.Matrix:
    return sp.Matrix([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def translation_matrix_3d(tx, ty, tz) -> sp.Matrix:
    return sp.Matrix([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])


def main() -> None:
    print("=== Homogeneous Coordinates + Translation Matrices Tool ===")
    print("Purpose: represent translation as matrix multiplication.")
    print("Type 'q' anytime to quit.\n")

    while True:
        try:
            print("Choose dimension:")
            print("1) 2D (x,y) -> homogeneous (x,y,1) with 3x3 matrices")
            print("2) 3D (x,y,z) -> homogeneous (x,y,z,1) with 4x4 matrices")
            choice = input("Choice (1/2): ").strip().lower()

            if choice in {"q", "quit", "exit"}:
                print("Bye!")
                return
            if choice not in {"1", "2"}:
                print("Invalid choice.")
                continue

            d = 2 if choice == "1" else 3

            p = read_point(d, f"Enter point p in R^{d} (e.g. 1, -2{' , 3' if d==3 else ''}): ")
            print("\np ="); sp.pprint(p, use_unicode=True)

            k = read_int("How many translations do you want to apply? (k): ", 1, 20)

            Ts = []
            for i in range(1, k + 1):
                t = read_point(d, f"Enter translation t{i} (e.g. 3, -1{' , 0' if d==3 else ''}): ")
                if d == 2:
                    T = translation_matrix_2d(t[0, 0], t[1, 0])
                else:
                    T = translation_matrix_3d(t[0, 0], t[1, 0], t[2, 0])
                Ts.append(T)
                print(f"\nT{i} ="); sp.pprint(T, use_unicode=True)

            # IMPORTANT: applying t1 then t2 means p' = T2*T1*p
            T_total = sp.eye(d + 1)
            for T in Ts:
                T_total = T * T_total

            ph = sp.Matrix(list(p) + [1])  # homogeneous point
            ph2 = sp.simplify(T_total * ph)
            p2 = from_homogeneous(ph2, d)

            print("\nHomogeneous point p~ ="); sp.pprint(ph, use_unicode=True)
            print("\nTotal translation matrix T_total ="); sp.pprint(T_total, use_unicode=True)
            print("\nResult homogeneous p~' = T_total * p~ ="); sp.pprint(ph2, use_unicode=True)
            print("\nResult point p' ="); sp.pprint(p2, use_unicode=True)

            # also show inverse quickly
            print("\nInverse translation (undo move) matrix T_total^{-1} =")
            sp.pprint(sp.simplify(T_total.inv()), use_unicode=True)

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            return
        except Exception as e:
            print("\n💥 Unexpected error (program continues):", repr(e))
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
