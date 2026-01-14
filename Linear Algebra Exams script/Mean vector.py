"""
hver enkelt vektor skal drejes fra vandret til lodret. Så vidst det siger v=(2,1,0).
Så skal den se sådan ud =
   [2]
v= [1]
   [0]

"""

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
        return sp.Rational(token)  # exact: supports -3, 1/2, 0.5
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def parse_row(row_text: str, n: int) -> list[sp.Rational]:
    s = row_text.strip().replace(",", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    if len(parts) != n:
        raise InputError(f"Expected {n} numbers, got {len(parts)}.")
    return [parse_number(p) for p in parts]


def read_int(prompt: str, min_value: int = 1, max_value: int = 200) -> int:
    while True:
        s = input(prompt).strip()
        if s.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if not s.isdigit():
            print("Please enter a positive integer.")
            continue
        v = int(s)
        if v < min_value or v > max_value:
            print(f"Please enter a number between {min_value} and {max_value}.")
            continue
        return v


def read_matrix(m: int, n: int, name: str) -> sp.Matrix:
    print(f"\nEnter matrix {name} row by row ({m} x {n}).")
    print("Each COLUMN is one data vector. Fractions allowed (e.g. 1/3).")
    rows: list[list[sp.Rational]] = []
    for i in range(m):
        while True:
            txt = input(f"{name} Row {i+1} (n={n}): ")
            try:
                rows.append(parse_row(txt, n))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 2, -3, 4/5")
    return sp.Matrix(rows)


def mean_of_columns(A: sp.Matrix) -> sp.Matrix:
    # sum columns: A * ones(n,1) gives an m×1 vector of column sums
    return sp.simplify((A * sp.ones(A.cols, 1)) / A.cols)


def main() -> None:
    print("=== Mean Vector of Columns Tool ===")
    print("Computes μ = (1/k) * (v1 + ... + vk) where v1..vk are the columns of A.")
    print("Type 'q' to quit.\n")

    while True:
        try:
            m = read_int("Vector dimension m (rows): ", 1, 200)
            k = read_int("Number of vectors k (columns): ", 1, 200)

            A = read_matrix(m, k, "A")
            print("\nA ="); sp.pprint(A, use_unicode=True)

            mu = mean_of_columns(A)
            print("\nMean vector μ (average of columns) =")
            sp.pprint(mu, use_unicode=True)

            # handy for copying
            print("\nAs entries:", [sp.simplify(mu[i, 0]) for i in range(mu.rows)])

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
