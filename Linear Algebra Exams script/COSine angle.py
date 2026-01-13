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


def read_int(prompt: str, min_value: int = 1, max_value: int = 500) -> int:
    while True:
        txt = input(prompt).strip()
        if txt.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not txt.isdigit():
            print("Please enter a positive integer.")
            continue
        n = int(txt)
        if n < min_value or n > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return n


def read_vector(n: int, name: str) -> sp.Matrix:
    """
    Accepts: "1, -2, 3" or "1 -2 3"
    Returns n×1 column vector.
    """
    while True:
        txt = input(f"Enter vector {name} with {n} entries: ").strip()
        if txt.lower() in {"q", "quit", "back"}:
            raise KeyboardInterrupt
        txt = txt.replace(",", " ")
        parts = [p for p in txt.split() if p.strip() != ""]
        if len(parts) != n:
            print(f"❌ Expected {n} numbers, got {len(parts)}.")
            print("   Example: 1, -2, 3, 4/5")
            continue
        try:
            nums = [parse_num(p) for p in parts]
            return sp.Matrix(nums)
        except InputError as e:
            print(f"❌ {e}")


def dot(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
    if a.shape != b.shape:
        raise InputError(f"Shape mismatch: a is {a.shape}, b is {b.shape}")
    return (a.T * b)[0]


def norm(v: sp.Matrix) -> sp.Expr:
    return sp.sqrt((v.T * v)[0])


def main() -> None:
    print("=== Angle / Cosine Similarity Tool ===")
    print("cos(theta) = (a·b) / (||a|| ||b||)")
    print("Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Vector length n: ", 1, 500)
            a = read_vector(n, "a")
            b = read_vector(n, "b")

            print("\na ="); sp.pprint(a, use_unicode=True)
            print("\nb ="); sp.pprint(b, use_unicode=True)

            da = sp.simplify(dot(a, b))
            na = sp.simplify(norm(a))
            nb = sp.simplify(norm(b))

            if na == 0 or nb == 0:
                raise InputError("Angle/cosine similarity is undefined if one vector is the zero vector.")

            cosv = sp.simplify(da / (na * nb))

            # Sometimes due to simplification, cosv can be slightly outside [-1,1] numerically.
            # We'll compute angle symbolically and also numeric.
            theta = sp.acos(cosv)
            theta_deg = sp.simplify(theta * 180 / sp.pi)

            print("\nDot product a·b =", da)
            print("||a|| =", na, "≈", sp.N(na))
            print("||b|| =", nb, "≈", sp.N(nb))
            print("\ncos(theta) =", cosv, "≈", sp.N(cosv))
            print("theta (rad) =", sp.simplify(theta), "≈", sp.N(theta))
            print("theta (deg) =", sp.simplify(theta_deg), "≈", sp.N(theta_deg))

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return
            print()

        except KeyboardInterrupt:
            print("\nBye!")
            return
        except InputError as e:
            print(f"\n⚠️ {e}")
            input("Press Enter to continue...")
        except Exception as e:
            print("\n💥 Unexpected error (program continues):", repr(e))
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
