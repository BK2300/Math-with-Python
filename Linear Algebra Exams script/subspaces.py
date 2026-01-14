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
        return sp.Rational(token)  # supports -3, 1/2, 0.5
    except Exception:
        raise InputError(f"Could not parse '{token}'. Use e.g. 3, -2, 1/4, 0.5")


def parse_row(row_text: str, n: int) -> list[sp.Rational]:
    s = row_text.strip()
    for ch in "[](){}":
        s = s.replace(ch, "")
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p.strip() != ""]
    if len(parts) != n:
        raise InputError(f"Expected {n} numbers, got {len(parts)}.")
    return [parse_number(p) for p in parts]


def read_int(prompt: str, min_value: int = 1, max_value: int = 80) -> int:
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


def read_matrix(m: int, n: int, name: str = "A") -> sp.Matrix:
    print(f"\nEnter matrix {name} row by row ({m} x {n}).")
    print("Separate numbers with spaces or commas. Fractions allowed (e.g. 1/3).")
    rows = []
    for i in range(m):
        while True:
            txt = input(f"{name} Row {i+1} (n={n}): ")
            if txt.lower() in {"q", "quit", "exit"}:
                raise KeyboardInterrupt
            try:
                rows.append(parse_row(txt, n))
                break
            except InputError as e:
                print(f"❌ {e}")
                print("Example row: 1, 2, -3, 4/5")
    return sp.Matrix(rows)


def nonzero_rows(M: sp.Matrix) -> list[sp.Matrix]:
    rows = []
    for i in range(M.rows):
        r = M.row(i)
        if any(r[0, j] != 0 for j in range(M.cols)):
            rows.append(r)
    return rows


def main() -> None:
    print("=== Subspaces / Bases Tool ===")
    print("Outputs bases for Col(A), Row(A), Null(A), plus rank and nullity.")
    print("Type 'q' to quit.\n")

    while True:
        try:
            m = read_int("Rows m: ", 1, 80)
            n = read_int("Cols n: ", 1, 80)
            A = read_matrix(m, n, "A")

            print("\nA =")
            sp.pprint(A, use_unicode=True)

            rrefA, pivots = A.rref()
            rankA = len(pivots)
            nullityA = n - rankA

            print("\nRREF(A) =")
            sp.pprint(rrefA, use_unicode=True)
            print("\nPivot columns (0-indexed):", pivots)
            print(f"rank(A) = {rankA}")
            print(f"nullity(A) = {nullityA}")
            print(f"rank + nullity = {rankA + nullityA} (should equal n = {n})")

            # Basis for Column Space: pivot columns of ORIGINAL A
            col_basis = [A.col(j) for j in pivots]
            print("\nBasis for Col(A) (pivot columns of ORIGINAL A):")
            if col_basis:
                for i, v in enumerate(col_basis, start=1):
                    print(f"v{i} =")
                    sp.pprint(v, use_unicode=True)
            else:
                print("{0} (only the zero vector)")

            # Basis for Row Space: nonzero rows of RREF(A)
            row_basis = nonzero_rows(rrefA)
            print("\nBasis for Row(A) (nonzero rows of RREF(A)):")
            if row_basis:
                for i, r in enumerate(row_basis, start=1):
                    print(f"r{i} =")
                    sp.pprint(r, use_unicode=True)
            else:
                print("{0} (only the zero vector)")

            # Basis for Null Space: nullspace vectors
            ns = A.nullspace()
            print("\nBasis for Null(A) (solutions to A*x = 0):")
            if ns:
                for i, v in enumerate(ns, start=1):
                    print(f"n{i} =")
                    sp.pprint(v, use_unicode=True)
            else:
                print("{0} (only the zero vector)")

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
