"""Positivt tal (>0): vektorerne peger nogenlunde samme vej (vinklen < 90°)
0: vektorerne er vinkelrette (90°) → kaldes ortogonale
Negativt tal (<0): vektorerne peger mere imod hinanden (vinklen > 90°)"""


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


def parse_number(token: str) -> sp.Rational:
    token = token.strip()
    if token == "":
        raise InputError("Empty number.")
    try:
        return sp.Rational(token)  # exact: supports 1/3, 0.5, -2
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def read_int(prompt: str, min_value: int = 1, max_value: int = 200) -> int:
    while True:
        s = input(prompt).strip()
        if s.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not s.isdigit():
            print("Please enter a positive integer.")
            continue
        val = int(s)
        if val < min_value or val > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return val


def read_vector(n: int, name: str) -> sp.Matrix:
    """
    Reads a vector of length n as a column vector (n x 1).
    Accepts commas or spaces, e.g. "1, 2, -3" or "1 2 -3".
    """
    while True:
        s = input(f"Enter vector {name} with {n} entries: ").strip()
        if s.lower() in {"q", "quit", "back"}:
            raise KeyboardInterrupt

        s = s.replace(",", " ")
        parts = [p for p in s.split() if p.strip() != ""]
        if len(parts) != n:
            print(f"❌ Expected {n} numbers, got {len(parts)}.")
            print("   Example: 1, 2, -3, 4/5")
            continue

        try:
            nums = [parse_number(p) for p in parts]
            return sp.Matrix(nums)  # column vector
        except InputError as e:
            print(f"❌ {e}")
            print("   Example: 1, 2, -3, 4/5")


def dot_product(a: sp.Matrix, b: sp.Matrix) -> sp.Rational:
    return (a.T * b)[0]


def main() -> None:
    print("=== Dot Product Tool ===")
    print("Dot product: a·b = sum(a_i * b_i)")
    print("Fractions are allowed. Type 'q' to quit.\n")

    while True:
        try:
            n = read_int("Vector length n: ", 1, 200)
            a = read_vector(n, "a")
            b = read_vector(n, "b")

            print("\na =")
            sp.pprint(a, use_unicode=True)
            print("\nb =")
            sp.pprint(b, use_unicode=True)

            dp = sp.simplify(dot_product(a, b))
            print(f"\na·b = {dp}")

            again = input("\nRun again? (Enter=yes, q=no): ").strip().lower()
            if again in {"q", "quit", "no", "n"}:
                print("Bye!")
                return

        except KeyboardInterrupt:
            print("\nBye!")
            return
        except Exception as e:
            print("\n💥 Unexpected error (program continues):", repr(e))
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
