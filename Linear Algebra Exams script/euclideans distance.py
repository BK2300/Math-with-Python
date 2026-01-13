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


def norm(v: sp.Matrix) -> sp.Expr:
    return sp.sqrt((v.T * v)[0])


def distance(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
    if a.shape != b.shape:
        raise InputError(f"Shape mismatch: a is {a.shape}, b is {b.shape}")
    return norm(a - b)


def main() -> None:
    print("=== Euclidean Distance Tool ===")
    print("Computes ||a-b|| and also shows a-b and its norm.")
    print("Fractions allowed. Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Vector length n: ", 1, 500)
            a = read_vector(n, "a")
            b = read_vector(n, "b")

            print("\na ="); sp.pprint(a, use_unicode=True)
            print("\nb ="); sp.pprint(b, use_unicode=True)

            diff = sp.simplify(a - b)
            d = sp.simplify(distance(a, b))

            print("\na - b ="); sp.pprint(diff, use_unicode=True)
            print(f"\n||a - b|| = {d}")
            print(f"≈ {sp.N(d)}")

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
